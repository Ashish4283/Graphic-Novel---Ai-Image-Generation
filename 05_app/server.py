"""
Graphic Novel Studio — a wrapper around ComfyUI for sequential panel production.

    pip install -r requirements.txt
    python server.py                 # http://127.0.0.1:8500

What this is for: a 200-page book is not 200 unrelated prompts. It is a cast
whose faces must not drift, environments that must stay the same shape as the
camera moves, and an art style that must hold from page 1 to page 200. This
app keeps that state — characters, scenes, panels — and builds a ComfyUI job
from it, so continuity comes from stored records rather than from remembering
what you typed yesterday.

ComfyUI itself does the rendering. This never reimplements it: panels are
submitted to ComfyUI's /prompt API and progress is read back from /history.

Binds to 127.0.0.1. On a cloud GPU reach it over an SSH tunnel:
    ssh -L 8500:localhost:8500 -L 8188:localhost:8188 root@<pod-ip>
"""

from __future__ import annotations

import json
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REFS = DATA / "refs"
OUTPUTS = DATA / "outputs"
PROJECT = DATA / "project.json"
TEMPLATES = HERE.parent / "03_workflows"

COMFY = "http://127.0.0.1:8188"

for d in (DATA, REFS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Graphic Novel Studio")


# ---------------------------------------------------------------- data model

def blank_project() -> dict:
    return {
        "name": "Untitled Graphic Novel",
        "style": {
            # The style block is applied to every panel. One place to change
            # it means page 200 cannot drift from page 1.
            "prompt": "graphic novel illustration, painterly, dramatic lighting",
            "negative": "photograph, 3d render, text, watermark, extra fingers",
            "checkpoint": "flux1-dev.safetensors",
            "style_lora": "",
            "style_lora_weight": 0.7,
            "steps": 25,
            "cfg": 3.5,
            "sampler": "euler",
            "scheduler": "simple",
            "width": 1024,
            "height": 1024,
            "controlnet_depth": "controlnet-depth-sdxl.safetensors",
            "controlnet_pose": "controlnet-openpose-sdxl.safetensors",
        },
        "characters": [],
        "scenes": [],
        "panels": [],
    }


def load() -> dict:
    if not PROJECT.is_file():
        save(blank_project())
    return json.loads(PROJECT.read_text(encoding="utf-8"))


def save(p: dict) -> None:
    tmp = PROJECT.with_suffix(".json.part")
    tmp.write_text(json.dumps(p, indent=2), encoding="utf-8")
    tmp.replace(PROJECT)          # atomic: a crash cannot truncate the project


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def find(items: list[dict], item_id: str) -> dict:
    for it in items:
        if it["id"] == item_id:
            return it
    raise HTTPException(404, f"not found: {item_id}")


# ---------------------------------------------------------------- schemas

class Style(BaseModel):
    prompt: str = ""
    negative: str = ""
    checkpoint: str = "flux1-dev.safetensors"
    style_lora: str = ""
    style_lora_weight: float = 0.7
    steps: int = 25
    cfg: float = 3.5
    sampler: str = "euler"
    scheduler: str = "simple"
    width: int = 1024
    height: int = 1024
    # Which ControlNet file to load. A scene with a depth reference uses the
    # depth model; one with only a pose reference uses the pose model.
    controlnet_depth: str = "controlnet-depth-sdxl.safetensors"
    controlnet_pose: str = "controlnet-openpose-sdxl.safetensors"


class Character(BaseModel):
    id: str | None = None
    name: str
    tier: int = Field(1, ge=1, le=2)      # 1 = trained LoRA, 2 = IPAdapter/PuLID
    variant: str = ""                      # "young", "older", "" for single-age
    trigger: str = ""
    lora_path: str = ""
    lora_weight: float = 0.85
    ref_images: list[str] = []
    appearance: str = ""                   # costume / distinguishing detail
    notes: str = ""


class Scene(BaseModel):
    id: str | None = None
    name: str
    environment: str = ""                  # locked description of the place
    seed: int = 0                          # 0 = random; fixed keeps a scene stable
    pose_ref: str = ""                     # OpenPose reference
    depth_ref: str = ""                    # Depth staging reference
    controlnet_strength: float = 0.7
    notes: str = ""


class Panel(BaseModel):
    id: str | None = None
    page: int = 1
    number: int = 1                        # panel order within the page
    scene_id: str = ""
    character_ids: list[str] = []
    action: str = ""                       # what happens in this panel
    camera: str = "medium shot"
    seed: int = 0
    status: str = "draft"                  # draft | queued | done | failed
    prompt_id: str = ""
    output: str = ""


# ---------------------------------------------------------------- prompt build

def compose_prompt(p: dict, panel: dict) -> dict:
    """
    Turn stored records into the text and conditioning for one panel.

    Order matters and is deliberate: style, then environment, then identity,
    then action, then camera. Identity tokens sit near the front of the
    character clause so they are not diluted by a long action description.
    """
    style = p["style"]
    scene = next((s for s in p["scenes"] if s["id"] == panel.get("scene_id")), None)
    chars = [c for c in p["characters"] if c["id"] in panel.get("character_ids", [])]

    parts = [style["prompt"]]
    if scene and scene.get("environment"):
        parts.append(scene["environment"])

    for c in chars:
        bits = []
        if c["tier"] == 1 and c.get("trigger"):
            bits.append(c["trigger"])
        bits.append(c["name"])
        if c.get("variant"):
            bits.append(f"{c['variant']} version")
        if c.get("appearance"):
            bits.append(c["appearance"])
        parts.append(", ".join(bits))

    if panel.get("action"):
        parts.append(panel["action"])
    if panel.get("camera"):
        parts.append(panel["camera"])

    tier1 = [c for c in chars if c["tier"] == 1 and c.get("lora_path")]
    tier2 = [c for c in chars if c["tier"] == 2]

    # Depth staging takes priority over pose when a scene supplies both: it
    # locks architecture, which is what panel-to-panel continuity needs.
    scene = scene or {}
    # Defaults cover projects saved before these fields existed, so an older
    # project.json does not silently lose its ControlNet.
    depth_model = style.get("controlnet_depth") or Style().controlnet_depth
    pose_model = style.get("controlnet_pose") or Style().controlnet_pose
    if scene.get("depth_ref"):
        cn_model, cn_image = depth_model, scene["depth_ref"]
    elif scene.get("pose_ref"):
        cn_model, cn_image = pose_model, scene["pose_ref"]
    else:
        cn_model, cn_image = "", ""

    return {
        "positive": ", ".join(x for x in parts if x),
        "negative": style["negative"],
        "loras": ([{"path": style["style_lora"], "weight": style["style_lora_weight"]}]
                  if style.get("style_lora") else [])
                 + [{"path": c["lora_path"], "weight": c["lora_weight"]} for c in tier1],
        "ipadapter_refs": [r for c in tier2 for r in c.get("ref_images", [])],
        "seed": panel.get("seed") or (scene or {}).get("seed") or 0,
        "controlnet": {
            "pose": scene.get("pose_ref", ""),
            "depth": scene.get("depth_ref", ""),
            "model": cn_model,
            "image": cn_image,
            "strength": scene.get("controlnet_strength", 0.7),
        },
        "width": style["width"],
        "height": style["height"],
        "steps": style["steps"],
        "cfg": style["cfg"],
        "sampler": style["sampler"],
        "scheduler": style["scheduler"],
        "checkpoint": style["checkpoint"],
        "warnings": _warn(chars, scene, tier1, tier2),
    }


def _warn(chars, scene, tier1, tier2) -> list[str]:
    """Surface continuity risks before GPU time is spent, not after."""
    w = []
    if not chars:
        w.append("no characters selected - the panel will be an empty environment")
    for c in chars:
        if c["tier"] == 1 and not c.get("lora_path"):
            w.append(f"{c['name']} is Tier 1 but has no LoRA path - identity will drift")
        if c["tier"] == 2 and not c.get("ref_images"):
            w.append(f"{c['name']} is Tier 2 but has no reference image")
    if scene is None:
        w.append("no scene - background continuity is not locked for this panel")
    elif not scene.get("seed"):
        w.append("scene seed is 0 (random) - panels in this scene will not match")
    if len(tier1) > 2:
        w.append(f"{len(tier1)} character LoRAs stacked - they compete; "
                 "consider regional prompting or lower weights")
    if len(tier2) > 3:
        w.append(f"{len(tier2)} IPAdapter references - faces tend to blend above ~3")
    return w


PLACEHOLDER = re.compile(r"%%[A-Z0-9_]+%%")

# A node is bypassed when the named input is empty. The map says which of its
# own inputs should replace each of its outputs, so consumers can be rewired
# to what fed it. This is the "modular bypass switch" in graph form: one
# template serves solo, multi-character and action panels instead of three.
BYPASS_RULES = {
    "LoraLoader": {
        "when_empty": "lora_name",
        "passthrough": {0: "model", 1: "clip"},
    },
    "ControlNetApplyAdvanced": {
        "when_empty": None,          # decided by its control image, see below
        "passthrough": {0: "positive", 1: "negative"},
    },
    "IPAdapterApply": {
        "when_empty": None,
        "passthrough": {0: "model"},
    },
}


def _rewire(wf: dict, dead_id: str, outputs: dict) -> None:
    """Point every consumer of dead_id's outputs at its upstream sources."""
    for node in wf.values():
        for key, value in list(node.get("inputs", {}).items()):
            if isinstance(value, list) and len(value) == 2 and value[0] == dead_id:
                replacement = outputs.get(value[1])
                if replacement is None:
                    raise HTTPException(
                        500,
                        f"cannot bypass node {dead_id}: output {value[1]} has no "
                        f"passthrough defined")
                node["inputs"][key] = replacement


def _prune_orphans(wf: dict) -> None:
    """Drop nodes nothing reaches from an output node (SaveImage/PreviewImage)."""
    sinks = [nid for nid, n in wf.items()
             if n.get("class_type") in ("SaveImage", "PreviewImage")]
    reachable: set[str] = set()
    stack = list(sinks)
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in wf:
            continue
        reachable.add(nid)
        for value in wf[nid].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                stack.append(value[0])
    for nid in [n for n in wf if n not in reachable]:
        del wf[nid]


def apply_bypasses(wf: dict, spec: dict) -> list[str]:
    """Remove nodes whose inputs are not supplied, rewiring around them."""
    bypassed: list[str] = []
    has_control = bool(spec["controlnet"]["model"] and spec["controlnet"]["image"])

    for nid, node in list(wf.items()):
        cls = node.get("class_type")
        rule = BYPASS_RULES.get(cls)
        if not rule:
            continue

        if cls == "LoraLoader":
            drop = not str(node["inputs"].get("lora_name", "")).strip()
        elif cls == "ControlNetApplyAdvanced":
            drop = not has_control
        elif cls == "IPAdapterApply":
            drop = not spec["ipadapter_refs"]
        else:
            drop = False
        if not drop:
            continue

        outputs = {idx: node["inputs"][src] for idx, src in rule["passthrough"].items()}
        _rewire(wf, nid, outputs)
        del wf[nid]
        bypassed.append(f"{cls}({nid})")

    _prune_orphans(wf)
    return bypassed


def build_workflow(spec: dict, template_name: str) -> tuple[dict, list[str]]:
    """
    Fill a ComfyUI API-format template with this panel's values, then bypass
    any node the panel does not supply inputs for.

    Templates use "%%PLACEHOLDER%%" strings so a graph exported from ComfyUI
    can be parameterised without this code needing to understand node wiring.
    """
    path = TEMPLATES / template_name
    if not path.is_file():
        raise HTTPException(400, f"workflow template not found: {template_name}")
    raw = path.read_text(encoding="utf-8")

    cn = spec["controlnet"]
    repl = {
        "%%POSITIVE%%": spec["positive"],
        "%%NEGATIVE%%": spec["negative"],
        "%%SEED%%": str(spec["seed"]),
        "%%STEPS%%": str(spec["steps"]),
        "%%CFG%%": str(spec["cfg"]),
        "%%SAMPLER%%": spec["sampler"],
        "%%SCHEDULER%%": spec["scheduler"],
        "%%WIDTH%%": str(spec["width"]),
        "%%HEIGHT%%": str(spec["height"]),
        "%%CHECKPOINT%%": spec["checkpoint"],
        "%%CONTROLNET%%": cn["model"],
        "%%CONTROL_IMAGE%%": cn["image"],
        "%%POSE_REF%%": cn["pose"],
        "%%DEPTH_REF%%": cn["depth"],
        "%%CN_STRENGTH%%": str(cn["strength"]),
        "%%CN_END%%": str(cn.get("end_percent", 0.5)),
        "%%LORA1%%": spec["loras"][0]["path"] if spec["loras"] else "",
        "%%LORA1_WEIGHT%%": str(spec["loras"][0]["weight"]) if spec["loras"] else "0",
        "%%IPADAPTER_REF%%": spec["ipadapter_refs"][0] if spec["ipadapter_refs"] else "",
    }
    for k, v in repl.items():
        raw = raw.replace(k, json.dumps(v)[1:-1])   # escape for JSON safety

    left = PLACEHOLDER.findall(raw)
    if left:
        raise HTTPException(
            500,
            f"template '{template_name}' has placeholders with no value: "
            f"{sorted(set(left))}")

    try:
        wf = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"template produced invalid JSON: {e}")

    wf.pop("_comment", None)
    bypassed = apply_bypasses(wf, spec)
    return wf, bypassed


# ---------------------------------------------------------------- routes

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/project")
def get_project() -> dict:
    return load()


@app.put("/api/project/name")
def set_name(body: dict) -> dict:
    p = load(); p["name"] = body.get("name", p["name"]); save(p)
    return {"ok": True}


@app.put("/api/style")
def set_style(style: Style) -> dict:
    p = load(); p["style"] = style.model_dump(); save(p)
    return {"ok": True}


# Two modes, same graph topology. Prototype proves the architecture on a small
# local card; production is what the client's book is actually rendered with.
PRESETS = {
    "prototype_sdxl": {
        "label": "Prototype — SDXL-Lightning (local, 4 GB card)",
        "template": "prototype_sdxl_api.json",
        "style": {
            "checkpoint": "sdxl_lightning_4step.safetensors",
            # Lightning is a 4-step model: more steps or any CFG above ~1.5
            # produces washed-out, overcooked panels.
            "steps": 4,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "sgm_uniform",
            "width": 1024,
            "height": 1024,
        },
        "notes": [
            "4 steps, CFG 1.0 — raising either breaks Lightning",
            "~40s per panel on a 4 GB card; use --lowvram",
            "Proves topology and continuity, not final quality",
        ],
    },
    "production_flux": {
        "label": "Production — FLUX.1 (rented 24 GB+ GPU)",
        "template": "solo_panel_api.json",
        "style": {
            "checkpoint": "flux1-dev.safetensors",
            "steps": 25,
            "cfg": 3.5,
            "sampler": "euler",
            "scheduler": "simple",
            "width": 1024,
            "height": 1024,
        },
        "notes": [
            "Needs 16-24 GB VRAM for inference, 24 GB+ for LoRA training",
            "FLUX.1 [dev] is NON-COMMERCIAL — confirm licensing with the client",
            "FLUX.1 [schnell] is Apache 2.0 if the book will be sold",
        ],
    },
}


@app.get("/api/presets")
def get_presets() -> dict:
    return {k: {"label": v["label"], "template": v["template"], "notes": v["notes"]}
            for k, v in PRESETS.items()}


@app.post("/api/presets/{name}")
def apply_preset(name: str) -> dict:
    """Switch sampler settings between prototype and production."""
    if name not in PRESETS:
        raise HTTPException(404, f"unknown preset: {name}")
    p = load()
    p["style"].update(PRESETS[name]["style"])
    p["mode"] = name
    save(p)
    return {"ok": True, "style": p["style"], "notes": PRESETS[name]["notes"],
            "template": PRESETS[name]["template"]}


def _upsert(kind: str, item: dict) -> dict:
    p = load()
    if item.get("id"):
        target = find(p[kind], item["id"])
        target.update(item)
    else:
        item["id"] = new_id(kind[:4])
        p[kind].append(item)
    save(p)
    return item


@app.post("/api/characters")
def upsert_character(c: Character) -> dict:
    return _upsert("characters", c.model_dump(exclude_none=False))


@app.post("/api/scenes")
def upsert_scene(s: Scene) -> dict:
    return _upsert("scenes", s.model_dump(exclude_none=False))


@app.post("/api/panels")
def upsert_panel(pn: Panel) -> dict:
    return _upsert("panels", pn.model_dump(exclude_none=False))


@app.delete("/api/{kind}/{item_id}")
def delete_item(kind: str, item_id: str) -> dict:
    if kind not in ("characters", "scenes", "panels"):
        raise HTTPException(400, "unknown collection")
    p = load()
    before = len(p[kind])
    p[kind] = [i for i in p[kind] if i["id"] != item_id]
    if len(p[kind]) == before:
        raise HTTPException(404, "not found")

    # Never leave a panel pointing at a deleted scene or character.
    if kind == "scenes":
        for pn in p["panels"]:
            if pn.get("scene_id") == item_id:
                pn["scene_id"] = ""
    if kind == "characters":
        for pn in p["panels"]:
            pn["character_ids"] = [c for c in pn.get("character_ids", []) if c != item_id]
    save(p)
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "ref.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "images only (.png, .jpg, .webp)")
    name = f"{uuid.uuid4().hex[:10]}{suffix}"
    (REFS / name).write_bytes(await file.read())
    return {"name": name, "url": f"/api/ref/{name}"}


@app.get("/api/ref/{name}")
def get_ref(name: str) -> FileResponse:
    path = (REFS / name).resolve()
    if REFS.resolve() != path.parent or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.get("/api/panels/{panel_id}/preview")
def preview(panel_id: str) -> dict:
    """The exact prompt and conditioning a panel would use. No GPU needed."""
    p = load()
    return compose_prompt(p, find(p["panels"], panel_id))


@app.get("/api/comfy/status")
async def comfy_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{COMFY}/system_stats")
            r.raise_for_status()
            return {"online": True, "stats": r.json()}
    except Exception as exc:                        # noqa: BLE001
        return {"online": False, "error": str(exc), "url": COMFY}


@app.post("/api/panels/{panel_id}/queue")
async def queue_panel(panel_id: str, body: dict | None = None) -> dict:
    """Submit one panel to ComfyUI."""
    template = (body or {}).get("template", "solo_panel_api.json")
    p = load()
    panel = find(p["panels"], panel_id)
    spec = compose_prompt(p, panel)
    workflow, bypassed = build_workflow(spec, template)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(f"{COMFY}/prompt", json={"prompt": workflow})
            r.raise_for_status()
            prompt_id = r.json().get("prompt_id", "")
    except Exception as exc:                        # noqa: BLE001
        panel["status"] = "failed"
        save(p)
        raise HTTPException(502, f"ComfyUI unreachable at {COMFY}: {exc}")

    panel["status"] = "queued"
    panel["prompt_id"] = prompt_id
    save(p)
    return {"ok": True, "prompt_id": prompt_id, "warnings": spec["warnings"],
            "bypassed": bypassed}


@app.post("/api/queue/batch")
async def queue_batch(body: dict) -> dict:
    """Queue many panels in page order — the batch generation the book needs."""
    ids = body.get("panel_ids") or []
    template = body.get("template", "solo_panel_api.json")
    results = []
    for pid in ids:
        try:
            r = await queue_panel(pid, {"template": template})
            results.append({"panel": pid, "ok": True, "prompt_id": r["prompt_id"]})
        except HTTPException as e:
            results.append({"panel": pid, "ok": False, "error": e.detail})
    return {"queued": sum(1 for r in results if r["ok"]), "results": results}


@app.get("/api/queue/status")
async def queue_status() -> dict:
    """Reconcile stored panel states against ComfyUI history."""
    p = load()
    pending = [x for x in p["panels"] if x.get("status") == "queued" and x.get("prompt_id")]
    if not pending:
        return {"updated": 0, "pending": 0}

    updated = 0
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            for panel in pending:
                r = await client.get(f"{COMFY}/history/{panel['prompt_id']}")
                if r.status_code != 200:
                    continue
                hist = r.json().get(panel["prompt_id"])
                if not hist:
                    continue
                images = [img for out in hist.get("outputs", {}).values()
                          for img in out.get("images", [])]
                if images:
                    panel["status"] = "done"
                    panel["output"] = images[0].get("filename", "")
                    updated += 1
    except Exception:                               # noqa: BLE001
        return {"updated": updated, "pending": len(pending), "comfy": "unreachable"}

    if updated:
        save(p)
    return {"updated": updated, "pending": len(pending) - updated}


@app.get("/api/templates")
def templates() -> dict:
    if not TEMPLATES.is_dir():
        return {"templates": []}
    return {"templates": sorted(f.name for f in TEMPLATES.glob("*_api.json"))}


@app.get("/api/stats")
def stats() -> dict:
    p = load()
    by_status: dict[str, int] = {}
    for pn in p["panels"]:
        by_status[pn.get("status", "draft")] = by_status.get(pn.get("status", "draft"), 0) + 1
    pages = sorted({pn.get("page", 1) for pn in p["panels"]})
    return {
        "characters": len(p["characters"]),
        "tier1": sum(1 for c in p["characters"] if c["tier"] == 1),
        "tier2": sum(1 for c in p["characters"] if c["tier"] == 2),
        "scenes": len(p["scenes"]),
        "panels": len(p["panels"]),
        "pages": len(pages),
        "by_status": by_status,
    }


if __name__ == "__main__":
    import argparse, uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--comfy", default=COMFY, help="ComfyUI base URL")
    a = ap.parse_args()
    COMFY = a.comfy
    print(f"Graphic Novel Studio  ->  http://{a.host}:{a.port}")
    print(f"ComfyUI expected at   ->  {COMFY}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")

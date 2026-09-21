"""
Continuity test — does ControlNet actually lock the environment across panels?

The claim being sold to the client is that backgrounds stay identical as a
scene unfolds. This measures it rather than asserting it.

Method: render the same scene three times with three different character
actions, twice over.
  Group A — WITHOUT ControlNet: text prompt alone
  Group B — WITH ControlNet depth, all three sharing one depth map

If the claim holds, Group B backgrounds are near-identical and Group A's are
not. Each render uses a DIFFERENT seed, so any similarity in Group B comes
from the depth map, not from reusing the same noise.

Temporary script.
"""
import json, shutil, sys, time
from pathlib import Path
import httpx

COMFY = "http://127.0.0.1:8188"
COMFY_DIR = Path(r"C:\ComfyUI")
OUT = COMFY_DIR / "output"
CONTROL_SRC = OUT / "panel_00001_.png"        # the establishing shot
CKPT = "sdxl_lightning_4step.safetensors"
CN_DEPTH = "controlnet-depth-sdxl.safetensors"

STYLE = ("biblical graphic novel illustration, painterly ink and wash, "
         "dramatic chiaroscuro lighting, muted earth palette")
SCENE = ("barren rocky summit, storm clouds boiling overhead, "
         "shafts of light breaking through")
NEG = "photograph, 3d render, text, watermark, extra fingers, blurry"

ACTIONS = [
    ("standing alone, arms raised toward the sky", "wide establishing shot"),
    ("kneeling with head bowed", "wide establishing shot"),
    ("walking forward holding a staff", "wide establishing shot"),
]
SEEDS = [111111, 222222, 333333]


def base_nodes(positive, seed):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
    }


def workflow_plain(positive, seed, prefix):
    n = base_nodes(positive, seed)
    n["11"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": prefix, "images": ["10", 0]}}
    return n


def workflow_controlled(positive, seed, prefix, control_image, strength=0.85):
    n = base_nodes(positive, seed)
    n["4"] = {"class_type": "LoadImage", "inputs": {"image": control_image}}
    n["5"] = {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["4", 0]}}
    n["6"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_DEPTH}}
    n["7"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["2", 0], "negative": ["3", 0], "control_net": ["6", 0],
        "image": ["5", 0], "strength": strength,
        "start_percent": 0.0, "end_percent": 0.9}}
    n["9"]["inputs"]["positive"] = ["7", 0]
    n["9"]["inputs"]["negative"] = ["7", 1]
    n["11"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": prefix, "images": ["10", 0]}}
    # also save the depth map once, so the client can see what locked the scene
    n["12"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "depthmap", "images": ["5", 0]}}
    return n


def run(wf, label):
    t0 = time.time()
    r = httpx.post(f"{COMFY}/prompt", json={"prompt": wf}, timeout=30)
    if r.status_code != 200:
        print(f"  {label}: SUBMIT FAILED {r.status_code} {r.text[:400]}")
        return None
    pid = r.json()["prompt_id"]
    for _ in range(150):
        time.sleep(3)
        h = httpx.get(f"{COMFY}/history/{pid}", timeout=15).json()
        if pid in h:
            entry = h[pid]
            imgs = [i for o in entry.get("outputs", {}).values() for i in o.get("images", [])]
            if imgs:
                names = [i["filename"] for i in imgs]
                print(f"  {label}: {time.time()-t0:5.0f}s  {names}")
                return names
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print(f"  {label}: ERROR {json.dumps(status)[:400]}")
                return None
    print(f"  {label}: timed out")
    return None


def main():
    if not CONTROL_SRC.is_file():
        sys.exit(f"missing establishing shot: {CONTROL_SRC}")
    inp = COMFY_DIR / "input"
    inp.mkdir(exist_ok=True)
    shutil.copy2(CONTROL_SRC, inp / "continuity_source.png")
    print(f"control source: {CONTROL_SRC.name} -> input/continuity_source.png\n")

    print("GROUP A — no ControlNet (text prompt only)")
    for i, ((action, camera), seed) in enumerate(zip(ACTIONS, SEEDS), 1):
        pos = f"{STYLE}, {SCENE}, an old bearded man in a brown robe, {action}, {camera}"
        run(workflow_plain(pos, seed, f"A_nocontrol_{i}"), f"A{i} {action[:34]}")

    print("\nGROUP B — ControlNet depth, all sharing ONE depth map")
    for i, ((action, camera), seed) in enumerate(zip(ACTIONS, SEEDS), 1):
        pos = f"{STYLE}, {SCENE}, an old bearded man in a brown robe, {action}, {camera}"
        run(workflow_controlled(pos, seed, f"B_control_{i}", "continuity_source.png"),
            f"B{i} {action[:34]}")

    print("\ndone")


if __name__ == "__main__":
    main()

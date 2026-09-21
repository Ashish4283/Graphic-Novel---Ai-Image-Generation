"""
Does an EMPTY-environment depth map free the character's pose?

The previous test showed that a depth map taken from a panel containing a
figure locks that figure's pose, and that lowering strength/end_percent does
not release it. This tests the ranked-first fix: stage from a plate with no
character in it.

  Step 1  render the location with no character   -> the "empty set"
  Step 2  two panels staged on its depth map, with opposite poses

If the fix works, the two panels share the location but differ in pose.
Temporary script.
"""
import time
from pathlib import Path
import httpx

COMFY = "http://127.0.0.1:8188"
COMFY_DIR = Path(r"C:\ComfyUI")
CKPT = "sdxl_lightning_4step.safetensors"
CN_DEPTH = "controlnet-depth-sdxl.safetensors"

STYLE = ("biblical graphic novel illustration, painterly ink and wash, "
         "dramatic chiaroscuro lighting, muted earth palette")
SCENE = "barren rocky summit, storm clouds boiling overhead, shafts of light"
NEG = "photograph, 3d render, text, watermark, blurry"
EMPTY_NEG = NEG + ", person, people, figure, man, character, silhouette"


def base(pos, seed, neg=NEG):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "sgm_uniform", "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
    }


def run(wf, label):
    t0 = time.time()
    r = httpx.post(f"{COMFY}/prompt", json={"prompt": wf}, timeout=30)
    if r.status_code != 200:
        print(f"  {label}: FAILED {r.text[:300]}"); return None
    pid = r.json()["prompt_id"]
    for _ in range(120):
        time.sleep(3)
        h = httpx.get(f"{COMFY}/history/{pid}", timeout=15).json()
        if pid in h and h[pid].get("outputs"):
            names = [i["filename"] for o in h[pid]["outputs"].values()
                     for i in o.get("images", [])]
            if names:
                print(f"  {label}: {time.time()-t0:4.0f}s  {names}")
                return names
    print(f"  {label}: timed out"); return None


print("STEP 1 - render the empty set (no character)")
wf = base(f"{STYLE}, {SCENE}, empty landscape, no people", 909090, EMPTY_NEG)
wf["11"] = {"class_type": "SaveImage",
            "inputs": {"filename_prefix": "emptyset", "images": ["10", 0]}}
run(wf, "empty set")

import shutil
src = sorted(COMFY_DIR.glob("output/emptyset_*.png"))[-1]
shutil.copy2(src, COMFY_DIR / "input" / "emptyset.png")
print(f"  staged: {src.name} -> input/emptyset.png")

print("\nSTEP 2 - two opposite poses on that empty-set depth map")
POSES = [("standing tall with both arms raised high", 171717),
         ("kneeling low with head bowed to the ground", 282828)]
for i, (action, seed) in enumerate(POSES, 1):
    pos = f"{STYLE}, {SCENE}, an old bearded man in a brown robe, {action}, wide shot"
    wf = base(pos, seed)
    wf["4"] = {"class_type": "LoadImage", "inputs": {"image": "emptyset.png"}}
    wf["5"] = {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["4", 0]}}
    wf["6"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_DEPTH}}
    wf["7"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["2", 0], "negative": ["3", 0], "control_net": ["6", 0],
        "image": ["5", 0], "strength": 0.55, "start_percent": 0.0, "end_percent": 0.5}}
    wf["9"]["inputs"]["positive"] = ["7", 0]
    wf["9"]["inputs"]["negative"] = ["7", 1]
    wf["11"] = {"class_type": "SaveImage",
                "inputs": {"filename_prefix": f"emptyset_pose{i}", "images": ["10", 0]}}
    run(wf, f"pose {i}: {action[:38]}")

print("\ndone")

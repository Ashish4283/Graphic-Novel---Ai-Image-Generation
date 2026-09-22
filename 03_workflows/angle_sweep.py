"""
Which IPAdapter setting actually lets the requested angle change?

The generator produced 13 front portraits labelled front/side/back. This asks
every candidate fix the same hard question: prompt for a BACK VIEW and see
whether anything turns around.

Measured, not eyeballed: perceptual distance from the seed image. A copy of
the seed scores near 0; a genuine back view should score high.
"""
import time
from pathlib import Path
import numpy as np
from PIL import Image
import httpx

COMFY = "http://127.0.0.1:8188"
OUT = Path(r"C:\ComfyUI\output")
SEED_IMG = "seed_char_d34ecbb8.png"          # the portrait seed already staged
CKPT = "RealVisXL_V5.0_Lightning.safetensors"
PLUS = "ip-adapter-plus_sdxl_vit-h.safetensors"
FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
CLIPV = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

POS = ("elderly man, long white beard, coarse brown robe, "
       "seen from directly behind, back view, facing away from camera, "
       "full body standing, even neutral lighting, plain grey background, "
       "cinematic photorealistic")
NEG = "front view, face visible, looking at camera, blurry, watermark"


def build(adapter, weight, start, end, seed, prefix, use_ip=True):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": POS, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 8, "cfg": 2.0, "sampler_name": "euler",
            "scheduler": "sgm_uniform", "denoise": 1.0, "model": ["1", 0],
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": prefix, "images": ["10", 0]}},
    }
    if use_ip:
        g["2"] = {"class_type": "LoadImage", "inputs": {"image": SEED_IMG, "upload": "image"}}
        g["3"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": adapter}}
        g["4"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIPV}}
        g["5"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["3", 0], "image": ["2", 0],
            "clip_vision": ["4", 0], "weight": weight, "weight_type": "linear",
            "combine_embeds": "concat", "start_at": start, "end_at": end,
            "embeds_scaling": "V only"}}
        g["9"]["inputs"]["model"] = ["5", 0]
    return g


def run(g, label):
    r = httpx.post(f"{COMFY}/prompt", json={"prompt": g}, timeout=30)
    if r.status_code != 200:
        print(f"  {label:26} SUBMIT {r.status_code} {r.text[:150]}"); return None
    pid = r.json()["prompt_id"]
    for _ in range(80):
        time.sleep(3)
        h = httpx.get(f"{COMFY}/history/{pid}", timeout=15).json()
        if pid in h and h[pid].get("outputs"):
            n = [i["filename"] for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            if n:
                return n[0]
    print(f"  {label:26} timed out"); return None


def arr(p, size=256):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)


seed_path = Path(r"C:\ComfyUI\input") / SEED_IMG
if not seed_path.is_file():
    raise SystemExit(f"seed not staged: {seed_path}")
seed_arr = arr(seed_path)

CASES = [
    ("no ipadapter (ceiling)", None, 0.0, 0.0, 0.0, False),
    ("plus w0.6 start0 end0.5", PLUS, 0.60, 0.0, 0.50, True),   # current default
    ("plus w0.6 start0.35",     PLUS, 0.60, 0.35, 0.90, True),
    ("plus w0.35 start0.2",     PLUS, 0.35, 0.20, 0.80, True),
    ("FACE w0.6 start0 end0.5", FACE, 0.60, 0.0, 0.50, True),
    ("FACE w0.7 start0.3",      FACE, 0.70, 0.30, 0.90, True),
]

print("prompt asks for a BACK VIEW. distance-from-seed: higher = actually changed\n")
results = []
for i, (label, ad, w, s, e, use_ip) in enumerate(CASES):
    tag = f"sweep/{i}_{label.replace(' ', '_').replace('.', '')}"
    f = run(build(ad, w, s, e, 4040 + i, tag, use_ip), label)
    if not f:
        continue
    d = float(np.abs(arr(OUT / "sweep" / f) - seed_arr).mean())
    results.append((label, d, f))
    print(f"  {label:26} distance {d:5.1f}   {f}")

print("\nranked:")
for label, d, f in sorted(results, key=lambda x: -x[1]):
    print(f"  {d:5.1f}  {label}")

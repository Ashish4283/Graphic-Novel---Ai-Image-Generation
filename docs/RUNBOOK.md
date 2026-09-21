# RUNBOOK — Graphic Novel Pipeline

How to run this project, locally and in the cloud.

- **Code:** `C:\Projects\comfyui-graphic-novel\`
- **Repo:** https://github.com/Ashish4283/Graphic-Novel---Ai-Image-Generation
- **ComfyUI:** `C:\ComfyUI\`
- **Shared models:** `C:\AI-Models\` — downloaded once, used by every project

---

## 1. The two modes

Same graph topology either way. Only the model and sampler settings change, so
whatever continuity behaviour is proven in prototype carries to production.

| | Prototype | Production |
|---|---|---|
| Model | SDXL-Lightning 4-step | FLUX.1 |
| Steps / CFG | **4 / 1.0** | 25 / 3.5 |
| Hardware | Local RTX 3050, 4 GB | Rented 24 GB+ |
| Template | `prototype_sdxl_api.json` | `solo_panel_api.json` |
| Purpose | Prove topology and continuity, free | Client-quality panels |

Switch in the app: **Style → Mode**.

> **Lightning breaks above 4 steps or CFG ~1.5.** Washed-out, overcooked panels
> are almost always someone raising these.

---

## 2. Start it up

### ComfyUI (the renderer)

```powershell
cd C:\ComfyUI
python main.py --lowvram --port 8188
```

`--lowvram` is required on a 4 GB card. On a rented 24 GB+ GPU, drop it.

### Studio (the production manager)

```powershell
cd C:\Projects\comfyui-graphic-novel\05_app
python server.py            # http://127.0.0.1:8500
```

The header pill shows whether ComfyUI is reachable. Both must run together:
Studio stores the book and builds the job; ComfyUI renders it.

On a cloud GPU, tunnel both rather than opening ports:

```bash
ssh -L 8500:localhost:8500 -L 8188:localhost:8188 root@<pod-ip>
```

---

## 3. Producing panels

1. **Style tab** — set the art style once. It is applied to every panel, which
   is what stops page 200 drifting from page 1.
2. **Characters** — Tier 1 for leads (trained LoRA, one per age variant),
   Tier 2 for supporting cast (reference images, no training).
3. **Scenes** — write the environment once and **fix the seed**. A scene with
   seed 0 renders a different room in every panel.
4. **Panels** — page, number, scene, cast, action, camera.
5. **Preview prompt** — see exactly what will be sent, and read the continuity
   warnings, before spending GPU time.
6. **Queue**, or **Queue all drafts** from the Dashboard for batch runs.
7. **Refresh status** — reconciles against ComfyUI history and pulls outputs.

### Continuity warnings, and why each matters

| Warning | Consequence if ignored |
|---|---|
| Tier 1 without a LoRA | The lead's face drifts between panels |
| Tier 2 without a reference | No identity conditioning at all |
| Scene seed is 0 | Backgrounds change shape mid-scene |
| More than 2 stacked LoRAs | Characters bleed into each other |
| More than ~3 IPAdapter refs | Faces blend into an average |
| No scene | Nothing locks the environment |

---

## 4. Dataset preparation (Phase 1)

Before any training:

```powershell
cd C:\Projects\comfyui-graphic-novel\01_dataset
python prepare_dataset.py --input refs/moses --trigger mosesv1 --check
python prepare_dataset.py --input refs/moses --trigger mosesv1 --out dataset/
```

Name files with the angle in them — `moses_front_01.png`, `moses_three_quarter_02.png`,
`moses_side_03.png`. Anything unrecognised is reported as unlabelled rather
than guessed at.

The validator refuses to build a failing dataset. That is deliberate: a LoRA
trained on soft, duplicated or angle-poor references cannot be fixed with
better prompts later — only by retraining.

---

## 5. Cloud

### Which GPU

| Instance | VRAM | Good for |
|---|---|---|
| RunPod RTX 4090 | 24 GB | Workflow iteration — cheapest per hour |
| AWS `g5.xlarge` (A10G) | 23 GB | FLUX inference; training is tight |
| AWS `g6e.xlarge` (L40S) | 46 GB | **FLUX LoRA training — comfortable** |

AWS costs 2.5–5× RunPod per hour, but the account holds **$100 of credits**
that expire unused. Spend credits on training; pay cash for iteration.

**The pending 8 vCPU G/VT quota covers all three AWS types above.** A100/H100
(P family) is a separate quota and is not needed.

### Cost per character LoRA

| | Per LoRA | 6 LoRAs |
|---|---|---|
| RTX 4090 @ $0.40/hr | ~$1.20 | ~$7 |
| A100 @ $1.40/hr | ~$4.20 | ~$25 |

Train on the 4090 or L40S unless 24 GB genuinely will not hold the run.

### Storage

FLUX + T5 + VAE + ControlNet + IPAdapter + PuLID is **60–80 GB**. Use a
persistent network volume (~$5/month) or re-download it every session.

---

## 6. Licensing — raise this with the client

**FLUX.1 [dev] is licensed for non-commercial use.** A graphic novel that will
be sold is a commercial use. Options:

- **FLUX.1 [schnell]** — Apache 2.0, commercial fine, 4-step, lower fidelity
- **A BFL commercial licence for [dev]** — exists, paid
- **SDXL** — permissive, mature LoRA/ControlNet/IPAdapter ecosystem, cheaper
  to train, and strong for stylised comic art

Settle this before training six FLUX LoRAs.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| ComfyUI won't start, `infer_schema ... list[int]` | torch older than ComfyUI expects | Upgrade torch (2.7+) |
| Studio says "ComfyUI offline" | Not running, or wrong port | Start it; `--comfy http://...` to point elsewhere |
| Queue returns 502 | ComfyUI unreachable | Panel is marked failed; fix and requeue |
| "workflow template not found" | Template missing from `03_workflows/` | Check the Dashboard dropdown |
| Washed-out panels | Lightning above 4 steps or CFG > ~1.5 | Reapply the prototype preset |
| CUDA out of memory | 4 GB too small for the node stack | `--lowvram`, smaller size, bypass ControlNet/IPAdapter |
| Characters drift between panels | Tier 1 without a LoRA, or no trigger token | Check the panel preview warnings |
| Background changes mid-scene | Scene seed is 0 | Set a fixed seed on the scene |
| GPU not visible (ASUS) | Armoury Crate Eco Mode | GPU Mode → Standard, then a true **Restart** |

---

## 8. Model library

Everything lives in `C:\AI-Models\`, registered via
`C:\ComfyUI\extra_model_paths.yaml`. Nothing is duplicated per project.

| Folder | Holds |
|---|---|
| `checkpoints/` | SDXL-Lightning, later FLUX |
| `loras/` | Trained character LoRAs |
| `controlnet/` | OpenPose, Depth |
| `ipadapter/`, `clip_vision/` | Reference-based identity |

On a rented pod, mount a network volume at the same layout and point
`extra_model_paths.yaml` at it — the setup is identical.

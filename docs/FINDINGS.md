# Findings — prototype validation

Measured results from running the pipeline, not estimates. All on an RTX 3050
Laptop GPU (4 GB) with SDXL-Lightning 4-step, `--lowvram`.

---

## 1. The stack runs on 4 GB

| | |
|---|---|
| Panel at 1024×1024, no ControlNet | **17–24 s** |
| Panel with ControlNet depth | **20–21 s** |
| First run of a session (model load) | +60–90 s |

ControlNet costs almost nothing in time and did **not** exhaust VRAM on a 4 GB
card. That was not a given — it's why the test was worth running.

## 2. ControlNet holds the environment — 42% measured reduction

Six renders: the same scene and three different character actions, once
without ControlNet and once with, all six on **different seeds** so any
similarity can only come from the depth map.

Mean absolute pixel difference between panels in a group (0–255, lower = more
alike):

| Group | Panel-to-panel | vs the establishing shot |
|---|---|---|
| A — text prompt only | **62.72** | 54.94 |
| B — shared depth map | **36.18** | 33.12 |

**ControlNet reduced panel-to-panel variation by 42%.** Visually the
depth-locked panels share the same rocky outcrop, the same flanking mountains
and the same composition; the uncontrolled panels share nothing but a mood.

Reproduce with `03_workflows/continuity_test.py` then `measure_continuity.py`.

## 3. Depth control at 0.85 also locks the *pose* — a real problem

Panel B2 was prompted "kneeling with head bowed" and rendered standing with
arms raised. The depth map was taken from a panel **containing a figure**, so
it constrained the character as well as the landscape.

For sequential art that is backwards: the set should be fixed and the actor
free. Fixes, most correct first:

1. **Take the depth map from an empty environment.** Render or draw the
   location with no character in it. The set, not the blocking.
2. **Lower `end_percent`** from 0.9 to ~0.5, so control shapes the early
   composition then releases and lets the prompt drive the subject.
3. **Lower `strength`** to 0.4–0.6.
4. **Regional conditioning** — depth over the background region, prompt over
   the character region. This is the "latent regional prompting" in the
   project pitch and is the production answer.

### 3a. Lowering strength and end_percent does NOT fix it — tested

Re-rendered at `strength 0.55`, `end_percent 0.5`, prompt "kneeling with head
bowed". Still standing, arms raised. The figure's silhouette is baked into the
depth map, so the model is not being nudged toward that pose — it is being
handed its shape. Fixes 2 and 3 are insufficient on their own.

### 3b. An empty-set depth map frees the pose but loses the character — tested

Rendered the location with no character, took depth from that, then staged two
opposite poses on it. The pose is no longer locked. But with no figure in the
depth map nothing reserves space for one, and both panels put tiny figures in
the far distance. Scale and placement became uncontrolled.

**Both failure modes are now bracketed:**

| Depth source | Environment | Character |
|---|---|---|
| Panel containing a figure | locked | pose locked too — cannot act |
| Empty environment | locked | scale and placement uncontrolled |

**Conclusion: depth alone cannot both lock the set and stage the actor.** One
signal cannot do two jobs.

**The production answer is two signals**, which is what the project pitch
already specifies — now derived from evidence rather than assumed:

- **Depth**, taken from an empty-set plate → architecture and perspective
- **OpenPose** stick figure → character size, position and pose

Both SDXL ControlNet models are installed. Wiring them together is the next
experiment, and it is the last open design question before FLUX.

### Production workflow this implies

Each scene is rendered **once** as an empty set. That plate becomes the
scene's permanent staging reference, and every panel in the scene is composed
inside it — exactly how a physical set works. This maps onto the Studio data
model already: `depth_ref` lives on the Scene, not the Panel.

## 4. Text identity alone is not enough

The first panel was prompted "wooden staff" with "arms raised" and produced a
figure holding **two** staffs. Costume and prop detail described in words
drifts between panels. This is the concrete argument for character LoRAs
(Tier 1) and IPAdapter references (Tier 2) rather than longer prompts.

---

## What this means for the build

- The architecture is validated end to end on free hardware.
- The remaining risk is **calibration**, not feasibility.
- The three settings that need tuning before any money is spent on a rented
  GPU: ControlNet `strength`, `end_percent`, and whether the depth source
  contains a character.

## Evidence

`docs/evidence/` holds the six renders and the depth map:

| File | |
|---|---|
| `panel_00001_.png` | Establishing shot; the depth map came from this |
| `depthmap_00001_.png` | What actually locked the scene |
| `A_nocontrol_1/2` | Text prompt only — backgrounds unrelated |
| `B_control_1/2` | Depth-locked — same location, different action |

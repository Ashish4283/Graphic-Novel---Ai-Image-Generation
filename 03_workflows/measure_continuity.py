"""Measure whether ControlNet actually held the environment. Temporary."""
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(r"C:\ComfyUI\output")
A = [OUT / f"A_nocontrol_{i}_00001_.png" for i in (1, 2, 3)]
B = [OUT / f"B_control_{i}_00001_.png" for i in (1, 2, 3)]


def arr(p, size=256):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)


def compare(paths, label):
    """Mean absolute pixel difference between every pair, 0-255."""
    imgs = [arr(p) for p in paths]
    diffs = []
    for (i, a), (j, b) in combinations(list(enumerate(imgs, 1)), 2):
        d = float(np.abs(a - b).mean())
        diffs.append(d)
        print(f"    panel {i} vs {j}: {d:6.2f}")
    mean = sum(diffs) / len(diffs)
    print(f"  {label} average difference: {mean:6.2f}")
    return mean


print("Lower = backgrounds more similar. Different seed in every render,")
print("so similarity can only come from the depth map.\n")

print("GROUP A - no ControlNet")
a = compare(A, "A")
print("\nGROUP B - ControlNet depth, one shared depth map")
b = compare(B, "B")

print(f"\n{'='*52}")
print(f"A (text only)      {a:6.2f}")
print(f"B (depth-locked)   {b:6.2f}")
if b < a:
    print(f"\nControlNet reduced panel-to-panel variation by {(1-b/a)*100:.0f}%")
else:
    print("\nNo improvement - ControlNet is not holding the scene")

# Also compare each B panel against the original establishing shot: the depth
# map came from it, so B panels should resemble it structurally.
src = OUT / "panel_00001_.png"
if src.is_file():
    s = arr(src)
    print("\ndifference from the establishing shot the depth map came from:")
    for name, group in (("A", A), ("B", B)):
        vals = [float(np.abs(arr(p) - s).mean()) for p in group]
        print(f"  {name}: {[f'{v:.1f}' for v in vals]}  avg {sum(vals)/len(vals):.2f}")

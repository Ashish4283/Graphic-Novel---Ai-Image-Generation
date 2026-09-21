"""
Phase 1 — turnaround dataset preparation for FLUX LoRA training.

Validates a character's reference sheet, reports every problem that would
damage the trained LoRA, and emits a Kohya/AI-Toolkit-ready training folder
with caption sidecars.

Why this phase gets the most care: a LoRA trained on inconsistent references
cannot be fixed by better prompts downstream. Blur, duplicates and missing
angles all show up later as face drift across panels, at which point the only
remedy is retraining. Catching them here costs minutes; catching them after
training costs GPU hours.

    python prepare_dataset.py --input refs/kai --trigger kaichr --check
    python prepare_dataset.py --input refs/kai --trigger kaichr --out dataset/

Angle is read from the filename: anything containing "front", "three_quarter"
(or "34"), "side"/"profile", "back", "detail". Unrecognised files are counted
as "unlabelled" and reported, never silently assigned.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Rules. Tuned for FLUX.1 [dev] LoRA training on character turnarounds.
# --------------------------------------------------------------------------

MIN_IMAGES = 20
MAX_IMAGES = 30
MIN_EDGE = 768            # FLUX trains at 1024; below 768 upscaling shows
IDEAL_EDGE = 1024
BLUR_FLOOR = 100.0        # variance of Laplacian; below this reads as soft
DUPLICATE_DISTANCE = 5    # Hamming distance on a 64-bit dHash
DARK_MEAN, BRIGHT_MEAN = 40, 215
LOW_CONTRAST_STD = 25

ANGLE_PATTERNS = {
    "front":         r"front|frontal|f_\d",
    "three_quarter": r"three[_-]?quarter|3[_-]?4|34|tq",
    "side":          r"side|profile|lateral",
    "back":          r"back|rear|behind",
    "detail":        r"detail|closeup|close[_-]up|face|head",
}
# A usable turnaround needs at least these, or the LoRA will not hold up when
# the camera swings around a character mid-scene.
REQUIRED_ANGLES = ("front", "three_quarter", "side")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# --------------------------------------------------------------------------

@dataclass
class ImageReport:
    path: str
    angle: str
    width: int = 0
    height: int = 0
    aspect: float = 0.0
    blur: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    dhash: str = ""
    issues: list[str] = field(default_factory=list)
    fatal: bool = False


def detect_angle(name: str) -> str:
    low = name.lower()
    for angle, pattern in ANGLE_PATTERNS.items():
        if re.search(pattern, low):
            return angle
    return "unlabelled"


def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash — near-duplicate detection, not cryptographic."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def laplacian_variance(img: Image.Image) -> float:
    """Sharpness proxy. Low variance = few edges = soft or out of focus."""
    g = np.asarray(img.convert("L"), dtype=np.float64)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # 'valid' convolution without scipy
    sub = np.lib.stride_tricks.sliding_window_view(g, (3, 3))
    lap = (sub * k).sum(axis=(-2, -1))
    return float(lap.var())


def inspect(path: Path) -> ImageReport:
    rep = ImageReport(path=path.name, angle=detect_angle(path.name))
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)     # honour camera rotation
            im.load()
            rep.width, rep.height = im.size
            rep.aspect = round(im.width / im.height, 3)
            rep.dhash = dhash(im)
            rep.blur = round(laplacian_variance(im), 1)
            g = np.asarray(im.convert("L"), dtype=np.float64)
            rep.mean, rep.std = round(float(g.mean()), 1), round(float(g.std()), 1)
    except Exception as exc:                      # noqa: BLE001
        rep.issues.append(f"unreadable: {exc}")
        rep.fatal = True
        return rep

    short = min(rep.width, rep.height)
    if short < MIN_EDGE:
        rep.issues.append(f"too small ({rep.width}x{rep.height}, need {MIN_EDGE}+ short edge)")
        rep.fatal = True
    elif short < IDEAL_EDGE:
        rep.issues.append(f"below ideal ({short}px short edge, {IDEAL_EDGE} preferred)")

    if rep.blur < BLUR_FLOOR:
        rep.issues.append(f"soft or out of focus (sharpness {rep.blur:.0f} < {BLUR_FLOOR:.0f})")
        rep.fatal = True
    if rep.mean < DARK_MEAN:
        rep.issues.append(f"very dark (mean {rep.mean})")
    if rep.mean > BRIGHT_MEAN:
        rep.issues.append(f"blown out (mean {rep.mean})")
    if rep.std < LOW_CONTRAST_STD:
        rep.issues.append(f"low contrast (std {rep.std})")
    if rep.angle == "unlabelled":
        rep.issues.append("angle not in filename — cannot verify turnaround coverage")

    return rep


def find_duplicates(reports: list[ImageReport]) -> list[tuple[str, str, int]]:
    """Near-duplicates waste training capacity and bias the LoRA."""
    pairs = []
    usable = [r for r in reports if r.dhash]
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            d = hamming(a.dhash, b.dhash)
            if d <= DUPLICATE_DISTANCE:
                pairs.append((a.path, b.path, d))
    return pairs


def audit(folder: Path) -> dict:
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not files:
        raise SystemExit(f"no images found in {folder}")

    reports = [inspect(p) for p in files]
    dupes = find_duplicates(reports)
    angles = Counter(r.angle for r in reports)
    usable = [r for r in reports if not r.fatal]

    problems: list[str] = []
    if len(usable) < MIN_IMAGES:
        problems.append(
            f"only {len(usable)} usable images; {MIN_IMAGES}-{MAX_IMAGES} recommended")
    if len(usable) > MAX_IMAGES:
        problems.append(
            f"{len(usable)} usable images; over {MAX_IMAGES} adds training time "
            "without adding identity information")
    for a in REQUIRED_ANGLES:
        if angles.get(a, 0) == 0:
            problems.append(f"no '{a}' reference — LoRA will be weak at that angle")
    if dupes:
        problems.append(f"{len(dupes)} near-duplicate pair(s)")

    aspects = Counter(r.aspect for r in usable)
    if len(aspects) > 4:
        problems.append(
            f"{len(aspects)} different aspect ratios; bucketing will crop "
            "inconsistently — consider standardising")

    return {
        "folder": str(folder),
        "total": len(reports),
        "usable": len(usable),
        "fatal": len(reports) - len(usable),
        "angles": dict(angles),
        "duplicates": dupes,
        "problems": problems,
        "images": [asdict(r) for r in reports],
    }


def print_audit(a: dict) -> bool:
    print(f"\n=== {a['folder']} ===")
    print(f"{a['total']} images  |  {a['usable']} usable  |  {a['fatal']} rejected\n")

    print("angle coverage")
    for angle in list(ANGLE_PATTERNS) + ["unlabelled"]:
        n = a["angles"].get(angle, 0)
        need = angle in REQUIRED_ANGLES
        mark = "ok " if n else ("MISSING" if need else "-  ")
        print(f"  {mark:8} {angle:<15} {n}")

    flagged = [i for i in a["images"] if i["issues"]]
    if flagged:
        print(f"\nper-image issues ({len(flagged)})")
        for i in flagged:
            tag = "REJECT" if i["fatal"] else "warn  "
            print(f"  {tag} {i['path']}")
            for msg in i["issues"]:
                print(f"         - {msg}")

    if a["duplicates"]:
        print("\nnear-duplicates")
        for x, y, d in a["duplicates"]:
            print(f"  distance {d}: {x}  ~  {y}")

    print("\nsummary")
    if a["problems"]:
        for p in a["problems"]:
            print(f"  ! {p}")
    else:
        print("  dataset passes every check")
    return not a["problems"]


def build(a: dict, src: Path, out: Path, trigger: str, repeats: int,
          caption_suffix: str) -> None:
    """Emit a Kohya-style '<repeats>_<trigger>' folder with caption sidecars."""
    target = out / f"{repeats}_{trigger}"
    target.mkdir(parents=True, exist_ok=True)

    written = 0
    for item in a["images"]:
        if item["fatal"]:
            continue
        s = src / item["path"]
        d = target / item["path"]
        shutil.copy2(s, d)

        # Caption names the angle so the LoRA learns identity separately from
        # pose. The trigger word carries identity; the rest stays flexible.
        parts = [trigger]
        angle = item["angle"]
        if angle == "front":
            parts.append("front view")
        elif angle == "three_quarter":
            parts.append("three-quarter view")
        elif angle == "side":
            parts.append("side profile")
        elif angle == "back":
            parts.append("back view")
        elif angle == "detail":
            parts.append("close-up portrait")
        if caption_suffix:
            parts.append(caption_suffix)
        d.with_suffix(".txt").write_text(", ".join(parts) + "\n", encoding="utf-8")
        written += 1

    (out / f"{trigger}_audit.json").write_text(
        json.dumps(a, indent=2), encoding="utf-8")

    print(f"\nwrote {written} images + captions -> {target}")
    print(f"audit report -> {out / (trigger + '_audit.json')}")
    print(f"\nKohya dataset root: {out}")
    print(f"  folder '{repeats}_{trigger}' means {repeats} repeats per epoch")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare a character turnaround dataset")
    ap.add_argument("--input", type=Path, required=True, help="folder of reference images")
    ap.add_argument("--trigger", required=True, help="unique trigger token, e.g. kaichr")
    ap.add_argument("--out", type=Path, help="dataset output root (omit for --check only)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--caption-suffix", default="",
                    help="appended to every caption, e.g. 'neutral lighting'")
    ap.add_argument("--check", action="store_true", help="audit only, write nothing")
    ap.add_argument("--force", action="store_true", help="build even if checks fail")
    args = ap.parse_args(argv)

    if not args.input.is_dir():
        raise SystemExit(f"not a folder: {args.input}")
    if re.search(r"\s", args.trigger):
        raise SystemExit("trigger must not contain spaces")

    a = audit(args.input)
    clean = print_audit(a)

    if args.check or not args.out:
        return 0 if clean else 1

    if not clean and not args.force:
        print("\nnot building: fix the problems above, or pass --force")
        return 1

    build(a, args.input, args.out, args.trigger, args.repeats, args.caption_suffix)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.acceptance.cubql_acceptance import compare_backend_results


def compare_query_results(old, new):
    same_capture = old.get("capture_sha256") == new.get("capture_sha256")
    same_mask = torch.equal(old["mask"], new["mask"])
    result = compare_backend_results(old, new)
    result.update({
        "same_capture": same_capture,
        "same_mask": same_mask,
        "passed": result["passed"] and same_capture and same_mask,
    })
    return result


def compare_atlases(old_file, new_file, output_dir):
    old = np.asarray(Image.open(old_file).convert("RGBA"), dtype=np.int16)
    new = np.asarray(Image.open(new_file).convert("RGBA"), dtype=np.int16)
    if old.shape != new.shape:
        raise ValueError(f"Atlas shapes differ: {old.shape} vs {new.shape}")
    delta = np.abs(old - new).astype(np.uint8)
    amplified = Image.fromarray(np.clip(delta[..., :3].astype(np.int16) * 8, 0, 255).astype(np.uint8))
    diff_file = output_dir / "baton-cubql-atlas-diff.png"
    preview_file = output_dir / "baton-cubql-comparison.png"
    amplified.save(diff_file)

    preview = Image.new("RGB", (1536, 536), (20, 24, 28))
    draw = ImageDraw.Draw(preview)
    for index, (label, image) in enumerate((
        ("old cubvh", Image.open(old_file).convert("RGB")),
        ("new cuBQL", Image.open(new_file).convert("RGB")),
        ("absolute diff x8", amplified),
    )):
        preview.paste(image.resize((512, 512), Image.Resampling.LANCZOS), (index * 512, 24))
        draw.text((index * 512 + 8, 6), label, fill=(235, 239, 243))
    preview.save(preview_file)
    return {
        "size": [old.shape[1], old.shape[0]],
        "max_channel_abs_diff": int(delta.max()),
        "mean_channel_abs_diff": float(delta.mean()),
        "changed_texels": int(np.any(delta != 0, axis=-1).sum()),
        "diff_file": str(diff_file),
        "preview_file": str(preview_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-atlas", type=Path, required=True)
    parser.add_argument("--new-atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old = torch.load(args.old, map_location="cpu", weights_only=False)
    new = torch.load(args.new, map_location="cpu", weights_only=False)
    report = compare_query_results(old, new)
    report["atlas"] = compare_atlases(args.old_atlas, args.new_atlas, args.output_dir)
    report_file = args.output_dir / "baton-cubql-comparison.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

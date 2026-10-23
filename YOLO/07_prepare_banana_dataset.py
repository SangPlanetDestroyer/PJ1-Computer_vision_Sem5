"""Normalize the banana dataset's mixed box and polygon labels for detection."""
import argparse
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ("unripe", "ripe", "overripe")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_annotation(line: str, path: Path, line_number: int) -> tuple[str, bool, bool, int]:
    parts = line.split()
    if not parts:
        raise ValueError(f"{path}:{line_number}: empty annotation")
    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: non-numeric annotation") from error

    if class_id not in range(len(CLASS_NAMES)):
        raise ValueError(f"{path}:{line_number}: unknown class id {class_id}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: non-finite coordinate")

    is_polygon = len(parts) >= 7 and len(parts) % 2 == 1
    if len(parts) == 5:
        center_x, center_y, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: box has no area")
        left, right = center_x - width / 2, center_x + width / 2
        top, bottom = center_y - height / 2, center_y + height / 2
    elif is_polygon:
        coordinates = list(zip(values[::2], values[1::2]))
        if len(coordinates) < 3:
            raise ValueError(f"{path}:{line_number}: polygon has fewer than 3 points")
        left = min(x for x, _ in coordinates)
        right = max(x for x, _ in coordinates)
        top = min(y for _, y in coordinates)
        bottom = max(y for _, y in coordinates)
    else:
        raise ValueError(f"{path}:{line_number}: expected 5 box fields or polygon coordinates")

    clipped_left = min(1.0, max(0.0, left))
    clipped_right = min(1.0, max(0.0, right))
    clipped_top = min(1.0, max(0.0, top))
    clipped_bottom = min(1.0, max(0.0, bottom))
    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        raise ValueError(f"{path}:{line_number}: box lies outside the image")

    was_clipped = any(
        not math.isclose(before, after)
        for before, after in (
            (left, clipped_left),
            (right, clipped_right),
            (top, clipped_top),
            (bottom, clipped_bottom),
        )
    )
    center_x = (clipped_left + clipped_right) / 2
    center_y = (clipped_top + clipped_bottom) / 2
    width = clipped_right - clipped_left
    height = clipped_bottom - clipped_top
    row = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    return row, is_polygon, was_clipped, class_id


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and os.path.samefile(source, destination):
        temporary = destination.with_name(f"{destination.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        return
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="datasets/Annotated Banana Ripeness Stage Dataset/BANANA_DATASET75",
    )
    parser.add_argument("--output", default="datasets/BANANA_DATASET75_DETECTION")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise SystemExit(f"Dataset source does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit("Source and output must be separate, non-nested directories")

    prepared: dict[str, list[tuple[Path, list[str]]]] = {}
    format_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    clipped_count = 0
    duplicate_count = 0

    for split in SPLITS:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem for path in images}
        if len(image_stems) != len(images) or image_stems != set(labels):
            raise SystemExit(f"{split}: image and label names do not match exactly")

        prepared[split] = []
        for image in images:
            label_path = labels[image.stem]
            normalized_rows = []
            seen_rows = set()
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row, is_polygon, was_clipped, class_id = parse_annotation(line, label_path, line_number)
                if row in seen_rows:
                    duplicate_count += 1
                    continue
                seen_rows.add(row)
                normalized_rows.append(row)
                format_counts["polygon" if is_polygon else "box"] += 1
                class_counts[class_id] += 1
                clipped_count += was_clipped
            prepared[split].append((image, normalized_rows))

    output.mkdir(parents=True, exist_ok=True)
    for split, records in prepared.items():
        for image, rows in records:
            link_image(image, output / "images" / split / image.name)
            label_path = output / "labels" / split / f"{image.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"{split}: {len(records)} images, {len(records)} labels")

    yaml = [
        f"path: {json.dumps(output.as_posix())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *(f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES)),
    ]
    (output / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"Source rows converted: {sum(format_counts.values())}")
    print(f"Source formats: {dict(format_counts)}")
    print(f"Boxes clipped to image bounds: {clipped_count}")
    print(f"Duplicate boxes removed: {duplicate_count}")
    print(f"Boxes by class: {dict(sorted(class_counts.items()))}")
    print(f"Detection dataset: {output}")


if __name__ == "__main__":
    main()
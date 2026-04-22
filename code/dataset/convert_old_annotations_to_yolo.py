from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


@dataclass
class LegacyObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class LegacyAnnotation:
    image_name: str
    width: int
    height: int
    objects: List[LegacyObject]


def yolo_xywh_from_xyxy(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width/height must be > 0")

    x1 = max(0.0, min(float(xmin), float(image_width)))
    y1 = max(0.0, min(float(ymin), float(image_height)))
    x2 = max(0.0, min(float(xmax), float(image_width)))
    y2 = max(0.0, min(float(ymax), float(image_height)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0

    return (
        cx / float(image_width),
        cy / float(image_height),
        box_w / float(image_width),
        box_h / float(image_height),
    )


def _required_text(elem: Optional[ET.Element], field_name: str) -> str:
    if elem is None or elem.text is None:
        raise ValueError(f"Missing required XML field: {field_name}")
    return elem.text.strip()


def parse_old_xml_annotation(annotation_path: Path) -> LegacyAnnotation:
    root = ET.parse(annotation_path).getroot()

    image_name = _required_text(root.find("filename"), "filename")
    size = root.find("size")
    if size is None:
        raise ValueError("Missing required XML field: size")

    width = int(_required_text(size.find("width"), "size.width"))
    height = int(_required_text(size.find("height"), "size.height"))

    objects: List[LegacyObject] = []
    for obj in root.findall("object"):
        name = _required_text(obj.find("name"), "object.name")
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(_required_text(box.find("xmin"), "object.bndbox.xmin"))
        ymin = float(_required_text(box.find("ymin"), "object.bndbox.ymin"))
        xmax = float(_required_text(box.find("xmax"), "object.bndbox.xmax"))
        ymax = float(_required_text(box.find("ymax"), "object.bndbox.ymax"))

        objects.append(
            LegacyObject(
                name=name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return LegacyAnnotation(image_name=image_name, width=width, height=height, objects=objects)


def parse_old_yaml_annotation(annotation_path: Path) -> LegacyAnnotation:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    image_name = str(data.get("image", annotation_path.with_suffix(".jpg").name))
    size = data.get("size") or {}
    width = int(size.get("width"))
    height = int(size.get("height"))

    objects: List[LegacyObject] = []
    for obj in data.get("objects", []):
        bndbox = obj.get("bndbox") or {}
        objects.append(
            LegacyObject(
                name=str(obj.get("name", "")),
                xmin=float(bndbox.get("xmin")),
                ymin=float(bndbox.get("ymin")),
                xmax=float(bndbox.get("xmax")),
                ymax=float(bndbox.get("ymax")),
            )
        )

    return LegacyAnnotation(image_name=image_name, width=width, height=height, objects=objects)


def parse_legacy_annotation(annotation_path: Path) -> LegacyAnnotation:
    suffix = annotation_path.suffix.lower()
    if suffix == ".xml":
        return parse_old_xml_annotation(annotation_path)
    if suffix in {".yaml", ".yml"}:
        return parse_old_yaml_annotation(annotation_path)
    raise ValueError(f"Unsupported annotation format: {annotation_path}")


def annotation_to_yolo_lines(
    annotation: LegacyAnnotation,
    class_map: Dict[str, int],
    default_class_id: Optional[int] = None,
    skip_unknown: bool = False,
) -> List[str]:
    lines: List[str] = []

    for obj in annotation.objects:
        if obj.name in class_map:
            class_id = class_map[obj.name]
        elif default_class_id is not None:
            class_id = default_class_id
        elif skip_unknown:
            continue
        else:
            raise KeyError(
                f"Class '{obj.name}' not found in class_map. "
                "Provide a mapping, default_class_id, or use skip_unknown=True."
            )

        x, y, w, h = yolo_xywh_from_xyxy(
            xmin=obj.xmin,
            ymin=obj.ymin,
            xmax=obj.xmax,
            ymax=obj.ymax,
            image_width=annotation.width,
            image_height=annotation.height,
        )
        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return lines


def convert_legacy_file_to_yolo(
    annotation_path: Path,
    output_label_path: Path,
    class_map: Dict[str, int],
    default_class_id: Optional[int] = None,
    skip_unknown: bool = False,
) -> int:
    annotation = parse_legacy_annotation(annotation_path)
    yolo_lines = annotation_to_yolo_lines(
        annotation,
        class_map=class_map,
        default_class_id=default_class_id,
        skip_unknown=skip_unknown,
    )

    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with output_label_path.open("w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")

    return len(yolo_lines)


def convert_annotations_one_by_one(
    annotation_paths: Sequence[Path],
    output_dir: Path,
    class_map: Dict[str, int],
    default_class_id: Optional[int] = None,
    skip_unknown: bool = False,
) -> List[Tuple[Path, Path, int]]:
    results: List[Tuple[Path, Path, int]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for annotation_path in annotation_paths:
        label_path = output_dir / f"{annotation_path.stem}.txt"
        n = convert_legacy_file_to_yolo(
            annotation_path=annotation_path,
            output_label_path=label_path,
            class_map=class_map,
            default_class_id=default_class_id,
            skip_unknown=skip_unknown,
        )
        results.append((annotation_path, label_path, n))

    return results


def _parse_class_map(class_map_items: Iterable[str]) -> Dict[str, int]:
    class_map: Dict[str, int] = {}
    for item in class_map_items:
        if "=" not in item:
            raise ValueError(f"Invalid --class-map value: '{item}'. Use name=id")
        name, class_id = item.split("=", 1)
        class_map[name.strip()] = int(class_id.strip())
    return class_map


def _collect_annotation_paths(inputs: Sequence[str], recursive: bool) -> List[Path]:
    supported_exts = {".xml", ".yaml", ".yml"}
    collected: List[Path] = []

    for raw in inputs:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in supported_exts:
            collected.append(p)
            continue

        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in p.glob(pattern):
                if child.is_file() and child.suffix.lower() in supported_exts:
                    collected.append(child)

    return sorted(set(collected))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy XML/YAML bbox annotations to YOLO labels."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input annotation files and/or directories",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where YOLO .txt label files are written",
    )
    parser.add_argument(
        "--class-map",
        action="append",
        default=[],
        help="Class mapping in name=id format, e.g. metal_ring=0",
    )
    parser.add_argument(
        "--default-class-id",
        type=int,
        default=None,
        help="Fallback class id for classes not in --class-map",
    )
    parser.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Skip objects whose class name is not in --class-map",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories in inputs",
    )
    args = parser.parse_args()

    class_map = _parse_class_map(args.class_map)
    annotation_paths = _collect_annotation_paths(args.inputs, recursive=args.recursive)

    if not annotation_paths:
        raise SystemExit("No supported annotation files found in inputs")

    results = convert_annotations_one_by_one(
        annotation_paths=annotation_paths,
        output_dir=Path(args.output_dir),
        class_map=class_map,
        default_class_id=args.default_class_id,
        skip_unknown=args.skip_unknown,
    )

    for annotation_path, label_path, n in results:
        print(f"{annotation_path} -> {label_path} ({n} objects)")


if __name__ == "__main__":
    main()

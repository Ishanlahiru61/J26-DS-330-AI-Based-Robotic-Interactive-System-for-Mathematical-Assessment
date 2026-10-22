

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any




@dataclass
class BoundingBox:
    """Pixel-coordinate oriented bounding box (OBB)."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float  # radians
    class_id: int = 0

    def to_yolo_obb(self, img_width: int, img_height: int) -> str:
        """Convert pixel OBB to YOLO normalized OBB format.

        YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        All values normalized to [0, 1].
        """
       
        dx = self.w / 2
        dy = self.h / 2
        
        corners = [
            (-dx, -dy),
            (dx, -dy),
            (dx, dy),
            (-dx, dy)
        ]
        
        
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        rotated_corners = []
        for x, y in corners:
            rx = self.cx + (x * cos_a - y * sin_a)
            ry = self.cy + (x * sin_a + y * cos_a)
            
            nx = max(0.0, min(1.0, rx / img_width))
            ny = max(0.0, min(1.0, ry / img_height))
            rotated_corners.extend([nx, ny])
            
        coords = " ".join(f"{c:.6f}" for c in rotated_corners)
        return f"{self.class_id} {coords}"


@dataclass
class AnnotationRecord:
    """Metadata for one annotated image."""

    filename: str
    split: str  
    img_width: int
    img_height: int
    boxes: list[dict[str, Any]] = field(default_factory=list)




CLASS_NAMES = {0: "cube"}
TRAIN_RATIO = 0.8


class DatasetManager:
    """Manage the YOLO dataset folder structure and metadata."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir)
        self.dataset = self.base / "dataset"
        self.images_train = self.dataset / "images" / "train"
        self.images_val = self.dataset / "images" / "val"
        self.labels_train = self.dataset / "labels" / "train"
        self.labels_val = self.dataset / "labels" / "val"
        self.metadata_file = self.dataset / "metadata.json"
        self._ensure_folders()
        self._metadata: list[dict[str, Any]] = self._load_metadata()

   

    def _ensure_folders(self) -> None:
        for folder in (
            self.images_train,
            self.images_val,
            self.labels_train,
            self.labels_val,
        ):
            folder.mkdir(parents=True, exist_ok=True)

 

    def _load_metadata(self) -> list[dict[str, Any]]:
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as fh:
                return json.load(fh)
        return []

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w") as fh:
            json.dump(self._metadata, fh, indent=2)



    def _next_index(self) -> int:
        """Return the next sequential image index."""
        if not self._metadata:
            return 1
        existing = []
        for record in self._metadata:
            name = record.get("filename", "")
            # Parse cube_NNNN from the filename
            stem = name.rsplit(".", 1)[0] if "." in name else name
            parts = stem.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                existing.append(int(parts[1]))
        return max(existing, default=0) + 1

    def _next_filename(self) -> str:
        idx = self._next_index()
        return f"cube_{idx:04d}"

    

    def _pick_split(self) -> str:
        """Round-robin train/val to approximate 80/20 split."""
        train_count = sum(1 for r in self._metadata if r["split"] == "train")
        val_count = sum(1 for r in self._metadata if r["split"] == "val")
        total = train_count + val_count
        if total == 0:
            return "train"
        current_train_ratio = train_count / total
        return "val" if current_train_ratio >= TRAIN_RATIO else "train"

    

    def save(
        self,
        image_bytes: bytes,
        boxes: list[dict[str, Any]],
        img_width: int,
        img_height: int,
    ) -> AnnotationRecord:
        """Save an image and its YOLO OBB labels to the dataset.

        Args:
            image_bytes: Raw JPEG bytes.
            boxes: List of dicts with keys cx, cy, w, h, angle, class_id.
            img_width: Original image width in pixels.
            img_height: Original image height in pixels.

        Returns:
            The saved AnnotationRecord.
        """
        basename = self._next_filename()
        split = self._pick_split()

        img_dir = self.images_train if split == "train" else self.images_val
        lbl_dir = self.labels_train if split == "train" else self.labels_val

        img_path = img_dir / f"{basename}.jpg"
        lbl_path = lbl_dir / f"{basename}.txt"

        
        with open(img_path, "wb") as fh:
            fh.write(image_bytes)

        
        yolo_lines = []
        for box_data in boxes:
           
            if "x_min" in box_data and "cx" not in box_data:
                cx = (box_data["x_min"] + box_data["x_max"]) / 2
                cy = (box_data["y_min"] + box_data["y_max"]) / 2
                w = box_data["x_max"] - box_data["x_min"]
                h = box_data["y_max"] - box_data["y_min"]
                angle = 0.0
            else:
                cx = float(box_data.get("cx", 0))
                cy = float(box_data.get("cy", 0))
                w = float(box_data.get("w", 0))
                h = float(box_data.get("h", 0))
                angle = float(box_data.get("angle", 0))

            bb = BoundingBox(
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                angle=angle,
                class_id=int(box_data.get("class_id", 0)),
            )
            yolo_lines.append(bb.to_yolo_obb(img_width, img_height))

        with open(lbl_path, "w") as fh:
            fh.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        
        normalized_boxes = []
        for box_data in boxes:
            if "x_min" in box_data and "cx" not in box_data:
                cx = (box_data["x_min"] + box_data["x_max"]) / 2
                cy = (box_data["y_min"] + box_data["y_max"]) / 2
                w = box_data["x_max"] - box_data["x_min"]
                h = box_data["y_max"] - box_data["y_min"]
                angle = 0.0
            else:
                cx = float(box_data.get("cx", 0))
                cy = float(box_data.get("cy", 0))
                w = float(box_data.get("w", 0))
                h = float(box_data.get("h", 0))
                angle = float(box_data.get("angle", 0))
                
            normalized_boxes.append({
                "cx": cx, "cy": cy, "w": w, "h": h, "angle": angle,
                "class_id": int(box_data.get("class_id", 0))
            })

        record = AnnotationRecord(
            filename=f"{basename}.jpg",
            split=split,
            img_width=img_width,
            img_height=img_height,
            boxes=normalized_boxes,
        )
        self._metadata.append(asdict(record))
        self._save_metadata()
        return record

    

    def list_annotations(self) -> list[dict[str, Any]]:
        """Return all annotation records."""
        return list(self._metadata)

    def get_annotation(self, filename: str) -> dict[str, Any] | None:
        """Get a single annotation record by image filename."""
        for record in self._metadata:
            if record["filename"] == filename:
                return record
        return None

    def delete_annotation(self, filename: str) -> bool:
        """Delete an image and its label from the dataset."""
        record = self.get_annotation(filename)
        if record is None:
            return False

        split = record["split"]
        stem = filename.rsplit(".", 1)[0]

        img_dir = self.images_train if split == "train" else self.images_val
        lbl_dir = self.labels_train if split == "train" else self.labels_val

        img_path = img_dir / filename
        lbl_path = lbl_dir / f"{stem}.txt"

        if img_path.exists():
            img_path.unlink()
        if lbl_path.exists():
            lbl_path.unlink()

        self._metadata = [r for r in self._metadata if r["filename"] != filename]
        self._save_metadata()
        return True

    def image_path(self, filename: str) -> Path | None:
        """Return the absolute path to a saved image."""
        record = self.get_annotation(filename)
        if record is None:
            return None
        split = record["split"]
        img_dir = self.images_train if split == "train" else self.images_val
        path = img_dir / filename
        return path if path.exists() else None

   

    def stats(self) -> dict[str, Any]:
        """Compute dataset statistics."""
        total_images = len(self._metadata)
        train_count = sum(1 for r in self._metadata if r["split"] == "train")
        val_count = sum(1 for r in self._metadata if r["split"] == "val")
        total_cubes = sum(len(r.get("boxes", [])) for r in self._metadata)
        avg_cubes = round(total_cubes / total_images, 1) if total_images else 0.0

        return {
            "total_images": total_images,
            "total_cubes": total_cubes,
            "avg_cubes_per_image": avg_cubes,
            "train_images": train_count,
            "val_images": val_count,
        }

    

    def export_yaml(self) -> str:
        """Generate data.yaml for YOLO training and write it to disk."""
        yaml_content = (
            "# YOLO dataset configuration\n"
            "# Auto-generated by Cube Dataset Collection App\n\n"
            f"path: {self.dataset.resolve()}\n\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n"
        )
        for class_id, name in CLASS_NAMES.items():
            yaml_content += f"  {class_id}: {name}\n"

        yaml_path = self.dataset / "data.yaml"
        with open(yaml_path, "w") as fh:
            fh.write(yaml_content)
        return yaml_content

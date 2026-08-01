#!/usr/bin/env python3

from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO


class BubbleDetector:

    def __init__(
        self,
        model_name="huyvux3005/manga109-segmentation-bubble/best.pt",
        confidence=0.25,
        image_size=1600,
        padding=10,
    ):

        print("Loading model...")

        self.model = YOLO(model_name)

        self.confidence = confidence
        self.image_size = image_size
        self.padding = padding

    def detect(self, image_path):

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        masks = None

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        for i, box in enumerate(result.boxes):

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detection = {
                "id": i,
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": float(box.conf[0]),
            }

            if masks is not None:

                mask = masks[i]

                detection["mask"] = mask

            detections.append(detection)

        detections.sort(key=lambda d: (d["y"], d["x"]))

        return detections

    def crop(self, image_path, output_dir):

        image = cv2.imread(str(image_path))

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        detections = self.detect(image_path)

        saved = []

        for bubble in detections:

            x = bubble["x"]
            y = bubble["y"]
            w = bubble["w"]
            h = bubble["h"]

            pad = self.padding

            x = max(0, x - pad)
            y = max(0, y - pad)

            w = min(image.shape[1] - x, w + pad * 2)
            h = min(image.shape[0] - y, h + pad * 2)

            crop = image[y:y + h, x:x + w]

            filename = output_dir / f"bubble_{bubble['id']:03d}.png"

            cv2.imwrite(str(filename), crop)

            bubble["file"] = filename.name

            if "mask" in bubble:
                del bubble["mask"]

            saved.append(bubble)

        return saved

    def crop_masks(self, image_path, output_dir):

        image = cv2.imread(str(image_path))

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        detections = self.detect(image_path)

        saved = []

        for bubble in detections:

            if "mask" not in bubble:
                continue

            mask = bubble["mask"]

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            mask = (mask * 255).astype(np.uint8)

            isolated = cv2.bitwise_and(
                image,
                image,
                mask=mask,
            )

            x = bubble["x"]
            y = bubble["y"]
            w = bubble["w"]
            h = bubble["h"]

            crop = isolated[y:y + h, x:x + w]

            filename = output_dir / f"bubble_{bubble['id']:03d}.png"

            cv2.imwrite(str(filename), crop)

            bubble["file"] = filename.name

            del bubble["mask"]

            saved.append(bubble)

        return saved

    def debug(self, image_path, output_file):

        image = cv2.imread(str(image_path))

        detections = self.detect(image_path)

        for bubble in detections:

            x = bubble["x"]
            y = bubble["y"]
            w = bubble["w"]
            h = bubble["h"]

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            label = f"{bubble['confidence']:.2f}"

            cv2.putText(
                image,
                label,
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(str(output_file), image)

    def save_json(self, detections, filename):

        clean = []

        for d in detections:

            d = dict(d)

            d.pop("mask", None)

            clean.append(d)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=4)


def process_folder(
    input_folder,
    output_folder,
):

    detector = BubbleDetector()

    input_folder = Path(input_folder)

    output_folder = Path(output_folder)

    output_folder.mkdir(exist_ok=True)

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    for image in sorted(input_folder.iterdir()):

        if image.suffix.lower() not in extensions:
            continue

        print(f"Processing {image.name}")

        page_dir = output_folder / image.stem

        page_dir.mkdir(exist_ok=True)

        detections = detector.crop_masks(
            image,
            page_dir,
        )

        detector.save_json(
            detections,
            page_dir / "bubbles.json",
        )

        detector.debug(
            image,
            page_dir / "debug.png",
        )

        print(f"Found {len(detections)} bubbles")


if __name__ == "__main__":

    process_folder(
        "pages",
        "output",
    )
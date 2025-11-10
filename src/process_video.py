import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from supervision import OverlapMetric
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Suppress the deprecation warning
warnings.filterwarnings("ignore", category=UserWarning)

from rfdetr import RFDETRLarge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a video with RF-DETR + ViT-S classification")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--rfdetr", required=False, help="Path to RF-DETR weights (.pth)", default="/mnt/DeepSea-AI/models/i2MAP/rfdetr-large-640x640/checkpoint_best_total.pth")
    parser.add_argument("--vits", required=False, help="Path to ViT-S model directory (HuggingFace format)", default="/mnt/DeepSea-AI/models/i2MAP/mbari-i2map-vits-b8-20251008/")
    parser.add_argument(
        "--slice",
        type=int,
        default=800,
        help="Slice width/height for InferenceSlicer (square). Default: 800",
    )
    parser.add_argument(
        "--output",
        help="Optional output video path. If omitted, will save next to input with _results suffix",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Detection confidence threshold for RF-DETR (default: 0.1)",
    )
    parser.add_argument(
        "--min_track_frames",
        type=int,
        default=3,
        help="ByteTrack minimum consecutive frames (default: 3)",
    )
    parser.add_argument(
        "--skip-vits",
        action="store_true",
        help="Skip ViT-S classification stage and use only RF-DETR detections",
    )
    parser.add_argument(
        "--class-agnostic",
        action="store_true",
        help="Collapse all classes to a single 'marine organism' class with id 0",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using device: mps")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using device: cuda")
        return torch.device("cuda")
    print("Using device: cpu")
    return torch.device("cpu")


def load_id_to_name_from_coco_json(rfdetr_weights_path: str) -> dict:
    weights_path = Path(rfdetr_weights_path).resolve()
    coco_json_path = weights_path.parent / "coco.json"
    if not coco_json_path.exists():
        raise FileNotFoundError(f"Expected coco.json next to RF-DETR weights at: {coco_json_path}")
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    return {cat["id"]: cat["name"] for cat in coco.get("categories", [])}


def crop_detections(d: sv.Detections, image: np.ndarray, square_dim: int = 224) -> list[np.ndarray]:
    """
    Crop detections to squares by padding the shorter dimension, then resize to square_dim x square_dim.
    Returns a list of cropped images.
    """
    h, w, _ = image.shape
    images = []
    for i in range(len(d)):
        x1, y1, x2, y2 = map(int, d.xyxy[i])

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        shorter_side = min(height, width)
        longer_side = max(height, width)
        delta = abs(longer_side - shorter_side)
        padding = delta // 2

        if width == shorter_side:
            x1 -= padding
            x2 += padding
        else:
            y1 -= padding
            y2 += padding

        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if y2 <= y1 or x2 <= x1:
            continue

        cropped_img = image[y1:y2, x1:x2, :]
        resized_img = cv2.resize(cropped_img, (square_dim, square_dim), interpolation=cv2.INTER_LINEAR)
        images.append(resized_img)

    return images


def classify_crops_with_vits(
    cropped_images: list[np.ndarray],
    vits_processor: AutoImageProcessor,
    vits_model: AutoModelForImageClassification,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Classify cropped images using ViT-S model.
    Returns (class_ids, confidences, class_names)
    """
    if len(cropped_images) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), []

    rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cropped_images]
    inputs = vits_processor(images=rgb_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vits_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        class_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        confidences = torch.max(probs, dim=-1).values.cpu().numpy()

    id2label = vits_model.config.id2label
    class_names = [id2label[int(i)] for i in class_ids]
    return class_ids, confidences, class_names


def main():
    args = parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else video_path.with_stem(video_path.stem + "_results")
    )

    device = resolve_device()

    # Load RF-DETR
    detr_model = RFDETRLarge(pretrain_weights=str(Path(args.rfdetr).resolve()))
    detr_model.model.model.to(device)
    #detr_model.optimize_for_inference()

    # COCO id->name mapping next to RF-DETR weights
    id_to_name_detr = load_id_to_name_from_coco_json(args.rfdetr)
    print(f"Loaded {len(id_to_name_detr)} DETR categories")

    # Load ViT-S (optional)
    vits_processor = None
    vits_model = None
    id_to_name_vits = None
    if not args.skip_vits:
        vits_dir = Path(args.vits).resolve()
        vits_processor = AutoImageProcessor.from_pretrained(str(vits_dir), return_tensors="pt")
        vits_model = AutoModelForImageClassification.from_pretrained(str(vits_dir)).to(device)
        id_to_name_vits = vits_model.config.id2label
        print(f"Loaded ViT-S model with {len(id_to_name_vits)} categories")
    else:
        print("Skipping ViT-S classification stage")

    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    tracker = sv.ByteTrack(minimum_consecutive_frames=args.min_track_frames)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def callback(image_slice: np.ndarray) -> sv.Detections:
        image_rgb = cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB)
        return detr_model.predict(image_rgb, threshold=args.threshold)

    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(args.slice, args.slice), overlap_metric=OverlapMetric.IOS)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = slicer(frame)
        print(f"Found {len(detections)} detections")

        if len(detections) > 0:
            # Remove edge detections
            h, w, _ = frame.shape
            margin = 5
            mask = (
                (detections.xyxy[:, 0] > margin)
                & (detections.xyxy[:, 1] > margin)
                & (detections.xyxy[:, 2] < (w - margin))
                & (detections.xyxy[:, 3] < (h - margin))
            )
            detections_kept = detections[mask]

            # Remove Physonectae class detections
            if len(detections_kept) > 0:
                num_before = len(detections_kept)
                physonectae_mask = np.array([
                    id_to_name_detr.get(int(class_id), "").lower() != "Physonectae"
                    for class_id in detections_kept.class_id
                ])
                detections_kept = detections_kept[physonectae_mask]
                num_after = len(detections_kept)
                print(f"Removed {num_before - num_after} Physonectae detections")

            # Remove overlapping detections
            detections_kept = detections_kept.with_nms(threshold=0.01, class_agnostic=True)

            # Track cleaned detections
            detections_kept = tracker.update_with_detections(detections_kept)

            # Apply class-agnostic mode if enabled
            if args.class_agnostic and len(detections_kept) > 0:
                detections_kept.class_id = np.zeros(len(detections_kept), dtype=np.int64)

            # Crop and classify only if ViT-S is enabled
            if vits_model is not None and not args.class_agnostic:
                cropped_images = crop_detections(detections_kept, frame)
                cls_ids, cls_confs, cls_names = classify_crops_with_vits(cropped_images, vits_processor, vits_model, device)

                # Replace detection classifications with ViTS model
                detections_kept.class_id = cls_ids
                detections_kept.confidence = cls_confs

                labels = [
                    f"ID:{tracker_id} {id_to_name_vits.get(int(class_id), 'Unknown')} {confidence:0.2f}"
                    for class_id, confidence, tracker_id in zip(
                        detections_kept.class_id, detections_kept.confidence, detections_kept.tracker_id
                    )
                ]
            elif args.class_agnostic:
                # Use class-agnostic label
                labels = [
                    f"ID:{tracker_id} marine organism {confidence:0.2f}"
                    for confidence, tracker_id in zip(
                        detections_kept.confidence, detections_kept.tracker_id
                    )
                ]
            else:
                # Use RF-DETR classifications
                labels = [
                    f"ID:{tracker_id} {id_to_name_detr.get(int(class_id), 'Unknown')} {confidence:0.2f}"
                    for class_id, confidence, tracker_id in zip(
                        detections_kept.class_id, detections_kept.confidence, detections_kept.tracker_id
                    )
                ]
            frame = box_annotator.annotate(scene=frame, detections=detections_kept)
            frame = label_annotator.annotate(scene=frame, detections=detections_kept, labels=labels)
            print(f"Tracked objects: {len(detections_kept)}")

        out.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Total frames processed {frame_count}. Output saved to {output_path}")


if __name__ == "__main__":
    main()
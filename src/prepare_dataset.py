#!/usr/bin/env python
"""
Dataset preparation utilities for RF-DETR training.

This module provides utilities to:
1. Convert VOC format datasets to COCO format
2. Split COCO datasets into train/valid/test sets
"""

import os
import shutil
import random
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter

try:
    from tqdm import tqdm
except ImportError:
    # Fallback no-op tqdm if the package is unavailable
    def tqdm(iterable, *args, **kwargs):
        return iterable


def split(original_coco, images_dirs, output_base, train_ratio=0.75, valid_ratio=0.20, test_ratio=0.05, seed=None):
    """
    Split a COCO dataset into train, validation, and test sets.

    Args:
        original_coco (str): Path to the original COCO JSON file.
        images_dirs (list): List of directories containing the images.
        output_base (str): Base directory for output splits (train/valid/test).
        train_ratio (float): Ratio of training data (default: 0.75).
        valid_ratio (float): Ratio of validation data (default: 0.20).
        test_ratio (float): Ratio of test data (default: 0.05).
        seed (int): Random seed for reproducibility (default: None).
    """
    if seed is not None:
        random.seed(seed)

    # Validate ratios
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + valid_ratio + test_ratio}")

    # Load COCO data
    with open(original_coco, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in tqdm(annotations, desc="Index annotations", unit="ann", leave=False):
        img_to_anns[ann["image_id"]].append(ann)

    # Shuffle image IDs
    image_ids = list(img_to_anns.keys())
    random.shuffle(image_ids)

    # Split into train, valid, test
    n = len(image_ids)
    train_end = int(train_ratio * n)
    valid_end = int((train_ratio + valid_ratio) * n)

    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:valid_end]
    test_ids = image_ids[valid_end:]

    splits = {"train": train_ids, "valid": val_ids, "test": test_ids}

    for split_name, ids in splits.items():
        if len(ids) == 0:
            print(f"Warning: {split_name} split is empty, skipping...")
            continue

        split_dir = os.path.join(output_base, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Filter images and annotations
        split_images = [img for img in images if img["id"] in ids]
        split_anns = []
        for iid in tqdm(ids, desc=f"Collect anns ({split_name})", unit="img", leave=False):
            split_anns.extend(img_to_anns[iid])

        # Remove any images (and their annotations) that don't have a corresponding file in images_dirs
        filtered_split_images = []
        for img in tqdm(split_images, desc=f"Filter images ({split_name})", unit="img", leave=False):
            file_found = False
            for images_dir in images_dirs:
                potential_src = os.path.join(images_dir, img["file_name"])
                if os.path.exists(potential_src):
                    file_found = True
                    break
            if file_found:
                filtered_split_images.append(img)
            else:
                print(
                    f"Info: Skipping image '{img['file_name']}' in '{split_name}' split because file not found in any images directory"
                )

        if len(filtered_split_images) != len(split_images):
            removed_count = len(split_images) - len(filtered_split_images)
            print(f"Removed {removed_count} images without files from '{split_name}' split")

        # Keep only annotations whose image_id remains after filtering
        kept_image_ids = {img["id"] for img in filtered_split_images}
        split_anns = [ann for ann in split_anns if ann["image_id"] in kept_image_ids]
        split_images = filtered_split_images

        # Create split COCO data
        split_data = {"images": split_images, "annotations": split_anns, "categories": categories}

        # Save annotations
        with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
            # Add info license fields to satisfy COCO format
            split_data["info"] = {
                "description": f"Dataset {split_name} split",
                "url": "",
                "version": "1.0",
                "year": 2025,
                "contributor": "",
                "date_created": "2024-01-01",
            }
            json.dump(split_data, f, indent=4)

        # Copy images from appropriate directories
        copied_count = 0
        for img in tqdm(split_images, desc=f"Copy images ({split_name})", unit="img", leave=False):
            src = None
            for images_dir in images_dirs:
                potential_src = os.path.join(images_dir, img["file_name"])
                if os.path.exists(potential_src):
                    src = potential_src
                    break
            if src:
                dst = os.path.join(split_dir, img["file_name"])
                shutil.copy(src, dst)
                copied_count += 1
            else:
                print(f"Warning: Image {img['file_name']} not found in any images directory")

        print(
            f"Split '{split_name}': {len(split_images)} images, {len(split_anns)} annotations, {copied_count} files copied"
        )

    print("Dataset split completed.")


def voc_to_coco(voc_datasets, output_json):
    """
    Combine multiple VOC datasets and convert to COCO format.

    Args:
        voc_datasets (list of tuples): List of (voc_dir, images_dir) tuples.
        output_json (str): Output path for COCO JSON.
    """
    # First, collect all class names from all XML files
    class_names = set()
    for voc_dir, images_dir in voc_datasets:
        xml_files = list(Path(voc_dir).glob("*.xml"))
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is not None:
                    name = name_elem.text
                    if name:
                        class_names.add(name)

    # Flag duplicates using lower case
    lower_names = [name.lower() for name in class_names]
    if len(lower_names) != len(set(lower_names)):
        print("Error: Duplicate categories found (case-insensitive):")
        counts = Counter(lower_names)
        for name_lower, count in counts.items():
            if count > 1:
                print(f"  {name_lower}: {count} occurrences")
        raise ValueError("Duplicate categories detected. Please resolve naming conflicts before proceeding.")

    # Use unique categories (preserving original case)
    categories = sorted(list(class_names))

    coco_data = {"images": [], "annotations": [], "categories": []}

    for i, cat in enumerate(categories):
        coco_data["categories"].append({"id": i, "name": cat, "supercategory": "none"})

    ann_id = 0
    img_id = 0
    img_filename_to_id = {}

    # Combine all XML files from all VOC datasets
    all_xml_files = []
    for voc_dir, images_dir in voc_datasets:
        all_xml_files.extend(list(Path(voc_dir).glob("*.xml")))

    for xml_file in all_xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Image info
        filename = root.find("filename").text
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        if filename not in img_filename_to_id:
            coco_data["images"].append({"id": img_id, "file_name": filename, "width": width, "height": height})
            img_filename_to_id[filename] = img_id
            img_id += 1
        image_id = img_filename_to_id[filename]

        # Annotations
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            name = name_elem.text
            if not name or name not in categories:
                continue
            cat_id = categories.index(name)
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            coco_data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(
        f"VOC to COCO conversion completed: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations, {len(coco_data['categories'])} categories"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset preparation utilities for RF-DETR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert VOC to COCO
  python prepare_dataset.py voc-to-coco \\
    --voc-dir /path/to/voc1 --images-dir /path/to/images1 \\
    --voc-dir /path/to/voc2 --images-dir /path/to/images2 \\
    --output output_coco.json

  # Split COCO dataset
  python prepare_dataset.py split \\
    --coco-json output_coco.json \\
    --images-dir /path/to/images1 --images-dir /path/to/images2 \\
    --output-dir /path/to/dataset \\
    --train-ratio 0.75 --valid-ratio 0.20 --test-ratio 0.05 
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # VOC to COCO conversion
    voc_parser = subparsers.add_parser("voc-to-coco", help="Convert VOC format to COCO format")
    voc_parser.add_argument(
        "--voc-dir", action="append", required=True, help="VOC annotation directory (can be specified multiple times)"
    )
    voc_parser.add_argument(
        "--images-dir",
        action="append",
        required=True,
        help="Images directory corresponding to each VOC directory (must match order)",
    )
    voc_parser.add_argument("--output", "-o", required=True, help="Output COCO JSON file path")

    # Split dataset
    split_parser = subparsers.add_parser("split", help="Split COCO dataset into train/valid/test")
    split_parser.add_argument("--coco-json", required=True, help="Input COCO JSON file")
    split_parser.add_argument(
        "--images-dir", action="append", required=True, help="Images directory (can be specified multiple times)"
    )
    split_parser.add_argument("--output-dir", "-o", required=True, help="Output directory for train/valid/test splits")
    split_parser.add_argument("--train-ratio", type=float, default=0.75, help="Training set ratio (default: 0.75)")
    split_parser.add_argument("--valid-ratio", type=float, default=0.20, help="Validation set ratio (default: 0.20)")
    split_parser.add_argument("--test-ratio", type=float, default=0.05, help="Test set ratio (default: 0.05)")
    split_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    split_parser.add_argument("--clean", action="store_true", help="Clean output directory before splitting")

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use --help for usage information.")
        return 1

    if args.command == "voc-to-coco":
        # Validate that voc-dir and images-dir have same length
        if len(args.voc_dir) != len(args.images_dir):
            print("Error: Number of --voc-dir and --images-dir arguments must match")
            return 1

        # Combine into list of tuples
        voc_datasets = list(zip(args.voc_dir, args.images_dir))
        voc_to_coco(voc_datasets, args.output)

    elif args.command == "split":
        # Clean output directory if requested
        if args.clean and os.path.exists(args.output_dir):
            print(f"Cleaning output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)

        split(
            args.coco_json,
            args.images_dir,
            args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

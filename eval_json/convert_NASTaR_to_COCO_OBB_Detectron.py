#!/usr/bin/env python3
"""
Convert NASTaR dataset (NovaSAR Automated Ship Target Recognition) to
COCO OBB format for Detectron2 / R-Sparse R-CNN.

NASTaR provides 512x512 SAR patches centred on ships with AIS metadata
(ship type, dimensions, heading). This script generates oriented bounding
box annotations from that metadata.

Usage:
    python convert_NASTaR_to_COCO_OBB_Detectron.py

Adjust DATASET_ROOT, OUTPUT_DIR, IMAGE_SUBDIR, CATEGORY_MODE, and
DATASET_PHASE below before running.

Written for use with R-Sparse R-CNN.
Contact: Adapted from Kamirul Kamirul's SSDD conversion script.
"""

import os
import csv
import json
import math
import time
import glob
import random
from collections import OrderedDict
from pathlib import Path

# ===========================================================================
# CONFIGURATION — adjust these paths and settings
# ===========================================================================

# Root of the extracted NASTaR dataset (the folder containing "Data/")
DATASET_ROOT = "./2tfa6x37oerz2lyiw6hp47058"

# Where to save the output JSON files
OUTPUT_DIR = "./eval_json"

# Which image subfolder to use:
#   "ship_patches"        — float32 GeoTIFF (original backscatter)
#   "ship_patches_uint8"  — uint8 GeoTIFF (for visualisation / training)
IMAGE_SUBDIR = "ship_patches_uint8"

# Category mode — choose how to label ships:
#   "single"   — single class "Ship" (id=1), best for pure detection
#   "multi_4"  — 4 classes: Fishing, Cargo, Tanker, Passenger
#   "multi_all"— all ship types present in the dataset
CATEGORY_MODE = "single"

# Dataset phase / split:
#   "all"           — use all patches
#   "offshore"      — only offshore patches (Dist_to_land > 500m)
#   "inshore"       — only inshore patches
#   "train"         — random 80% split
#   "test"          — random 20% split
DATASET_PHASE = "train"

# Train/test split ratio (only used when DATASET_PHASE is "train" or "test")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Image dimensions (NASTaR patches are all 512x512)
IMG_WIDTH = 512
IMG_HEIGHT = 512

# Pixel resolution in metres (from GeoTIFF metadata, ~2.5 m/pixel at 6m resolution)
# This is approximate; for more accuracy, read each GeoTIFF's transform.
# The paper states 512 pixels = 1280m, so 2.5 m/pixel.
PIXEL_SIZE_M = 2.5

# Minimum bounding box size in pixels (skip ships smaller than this)
MIN_BBOX_SIZE_PX = 3.0

# ===========================================================================
# CATEGORY DEFINITIONS
# ===========================================================================

CATEGORIES_SINGLE = [
    {"id": 1, "name": "Ship", "supercategory": "Ship"}
]

CATEGORIES_MULTI_4 = [
    {"id": 1, "name": "Fishing", "supercategory": "Ship"},
    {"id": 2, "name": "Cargo", "supercategory": "Ship"},
    {"id": 3, "name": "Tanker", "supercategory": "Ship"},
    {"id": 4, "name": "Passenger", "supercategory": "Ship"},
]

# Will be populated dynamically from the dataset
CATEGORIES_MULTI_ALL = []

# Map from AIS "Ship type" string to category for multi_4 mode
SHIP_TYPE_TO_MULTI4 = {
    "Fishing": 1,
    "Cargo": 2,
    "Tanker": 3,
    "Passenger": 4,
}

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================


def normalise_angle(angle_deg):
    """Normalise angle to [-180, 180) range."""
    while angle_deg >= 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def ais_heading_to_detectron_angle(heading_deg):
    """
    Convert AIS heading to Detectron2 RotatedBoxes angle convention.

    AIS heading: 0-360 degrees, 0=North, clockwise.
    Detectron2: degrees, counter-clockwise from positive x-axis.
                When angle=0, width is along x-axis, height along y-axis.

    For a ship pointing North (heading=0), the long axis (height/length)
    is along the y-axis, which corresponds to Detectron2 angle=0.

    For heading=90 (East), the ship is rotated 90 degrees CW from North,
    which is -90 degrees in Detectron2's CCW convention.

    Therefore: detectron_angle = -heading
    """
    angle = -heading_deg
    return normalise_angle(angle)


def parse_float_safe(value, default=None):
    """Safely parse a float from a CSV value, returning default if empty/invalid."""
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def collect_all_ais_records(dataset_root):
    """
    Walk through all scene folders and collect AIS records with their
    corresponding image paths.
    """
    data_dir = os.path.join(dataset_root, "Data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Make sure DATASET_ROOT points to the extracted NASTaR folder."
        )

    records = []
    scene_dirs = sorted(glob.glob(os.path.join(data_dir, "NovaSAR_*")))

    for scene_dir in scene_dirs:
        ais_csv = os.path.join(scene_dir, "AIS.csv")
        if not os.path.isfile(ais_csv):
            print(f"  [WARN] No AIS.csv in {scene_dir}, skipping.")
            continue

        scene_name = os.path.basename(scene_dir)
        img_dir = os.path.join(scene_dir, IMAGE_SUBDIR)

        if not os.path.isdir(img_dir):
            print(f"  [WARN] No {IMAGE_SUBDIR}/ in {scene_dir}, skipping.")
            continue

        # Read AIS.csv — first line is a comment starting with #
        with open(ais_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove leading # from header
        if lines and lines[0].startswith("#"):
            lines[0] = lines[0].lstrip("# ")

        reader = csv.DictReader(lines)

        for row in reader:
            patch_name = row.get("Patch_name", "").strip()
            if not patch_name:
                continue

            # Find the corresponding image file
            # In ship_patches_uint8/, files have "_uint8" appended:
            #   AIS Patch_name: NovaSAR_..._patch_1_Fishing
            #   Actual file:    NovaSAR_..._patch_1_Fishing_uint8.tif
            # In ship_patches/, files match the Patch_name directly.
            img_path = os.path.join(img_dir, patch_name + ".tif")
            if not os.path.isfile(img_path):
                # Try with _uint8 suffix (for ship_patches_uint8 folder)
                img_path = os.path.join(img_dir, patch_name + "_uint8.tif")
            if not os.path.isfile(img_path):
                # Try glob for any matching pattern (handles spaces etc.)
                candidates = glob.glob(
                    os.path.join(img_dir, patch_name + "*")
                )
                if candidates:
                    img_path = candidates[0]
                else:
                    continue

            # Extract relevant fields
            ship_type = row.get("Ship type", "Unknown").strip()
            width_m = parse_float_safe(row.get("Width"))
            length_m = parse_float_safe(row.get("Length"))
            heading = parse_float_safe(row.get("Heading"))
            dist_to_land = parse_float_safe(row.get("Dist_to_land", "0"))
            shoreline = row.get("Shoreline", "").strip()

            # Build relative image path (relative to the dataset root)
            rel_img_path = os.path.relpath(img_path, dataset_root)

            records.append({
                "patch_name": patch_name,
                "image_path": img_path,
                "rel_image_path": rel_img_path,
                "ship_type": ship_type,
                "width_m": width_m,
                "length_m": length_m,
                "heading": heading,
                "dist_to_land": dist_to_land,
                "shoreline": shoreline,
                "scene_name": scene_name,
            })

    return records


def filter_records(records, phase):
    """Filter records based on the dataset phase/split."""
    if phase == "all":
        return records
    elif phase == "offshore":
        return [r for r in records if r["shoreline"] == "offshore"]
    elif phase == "inshore":
        return [r for r in records if r["shoreline"] == "inshore"]
    elif phase in ("train", "test"):
        # Deterministic random split
        random.seed(RANDOM_SEED)
        indices = list(range(len(records)))
        random.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_RATIO)
        if phase == "train":
            return [records[i] for i in indices[:split_idx]]
        else:
            return [records[i] for i in indices[split_idx:]]
    else:
        raise ValueError(f"Unknown DATASET_PHASE: {phase}")


def get_categories(records, mode):
    """Get category list and type-to-id mapping based on mode."""
    if mode == "single":
        categories = CATEGORIES_SINGLE
        type_to_id = None  # Everything maps to 1
        return categories, type_to_id

    elif mode == "multi_4":
        categories = CATEGORIES_MULTI_4
        type_to_id = SHIP_TYPE_TO_MULTI4
        return categories, type_to_id

    elif mode == "multi_all":
        # Collect all unique ship types
        all_types = sorted(set(r["ship_type"] for r in records))
        categories = []
        type_to_id = {}
        for i, t in enumerate(all_types, start=1):
            categories.append({
                "id": i,
                "name": t,
                "supercategory": "Ship"
            })
            type_to_id[t] = i
        return categories, type_to_id

    else:
        raise ValueError(f"Unknown CATEGORY_MODE: {mode}")


def build_coco_json(records, categories, type_to_id, phase):
    """Build the COCO OBB JSON structure."""
    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    skipped_no_dims = 0
    skipped_no_heading = 0
    skipped_no_category = 0
    skipped_too_small = 0

    for rec in records:
        # Check if we have valid dimensions
        width_m = rec["width_m"]
        length_m = rec["length_m"]
        heading = rec["heading"]

        if width_m is None or length_m is None:
            skipped_no_dims += 1
            continue

        if heading is None:
            skipped_no_heading += 1
            continue

        # Determine category
        if type_to_id is None:
            # Single class mode
            cat_id = 1
        else:
            cat_id = type_to_id.get(rec["ship_type"])
            if cat_id is None:
                skipped_no_category += 1
                continue

        # Convert dimensions to pixels
        w_px = width_m / PIXEL_SIZE_M
        h_px = length_m / PIXEL_SIZE_M

        # Skip tiny ships
        if w_px < MIN_BBOX_SIZE_PX and h_px < MIN_BBOX_SIZE_PX:
            skipped_too_small += 1
            continue

        # Ensure h >= w (convention: h is the longer dimension = ship length)
        if w_px > h_px:
            w_px, h_px = h_px, w_px

        # Convert heading to Detectron2 angle
        angle_deg = ais_heading_to_detectron_angle(heading)

        # Ship is centred in the 512x512 patch
        cx = IMG_WIDTH / 2.0
        cy = IMG_HEIGHT / 2.0

        # Compute area
        area = w_px * h_px

        image_id += 1
        ann_id += 1

        # Add image entry
        images.append({
            "height": IMG_HEIGHT,
            "width": IMG_WIDTH,
            "id": image_id,
            "file_name": rec["rel_image_path"],
        })

        # Add annotation entry
        annotations.append({
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [cx, cy, w_px, h_px, angle_deg],
            "segmentation": [],
            "category_id": cat_id,
            "id": ann_id,
            "area": area,
        })

    # Build the full COCO structure
    today_str = time.strftime("%Y-%m-%d")
    current_year = int(time.strftime("%Y"))

    coco_json = OrderedDict()
    coco_json["info"] = {
        "description": f"NASTaR COCO OBB ({phase})",
        "version": "1.0",
        "year": current_year,
        "contributor": "Auto-generated from NASTaR AIS metadata",
        "date_created": today_str,
    }
    coco_json["images"] = images
    coco_json["annotations"] = annotations
    coco_json["categories"] = categories

    print(f"\n  Images:              {len(images)}")
    print(f"  Annotations:         {len(annotations)}")
    print(f"  Skipped (no dims):   {skipped_no_dims}")
    print(f"  Skipped (no heading):{skipped_no_heading}")
    print(f"  Skipped (no cat):    {skipped_no_category}")
    print(f"  Skipped (too small): {skipped_too_small}")

    return coco_json


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("NASTaR → COCO OBB Conversion for Detectron2 / R-Sparse R-CNN")
    print("=" * 60)

    print(f"\nDataset root:  {DATASET_ROOT}")
    print(f"Image subdir:  {IMAGE_SUBDIR}")
    print(f"Category mode: {CATEGORY_MODE}")
    print(f"Dataset phase: {DATASET_PHASE}")
    print(f"Pixel size:    {PIXEL_SIZE_M} m/pixel")

    # Step 1: Collect all AIS records
    print("\n[1/4] Scanning AIS.csv files...")
    records = collect_all_ais_records(DATASET_ROOT)
    print(f"  Found {len(records)} total records across all scenes.")

    # Step 2: Filter by phase
    print(f"\n[2/4] Filtering for phase '{DATASET_PHASE}'...")
    filtered = filter_records(records, DATASET_PHASE)
    print(f"  {len(filtered)} records after filtering.")

    # Step 3: Determine categories
    print(f"\n[3/4] Setting up categories ({CATEGORY_MODE})...")
    categories, type_to_id = get_categories(filtered, CATEGORY_MODE)
    for cat in categories:
        print(f"  {cat['id']}: {cat['name']}")

    # Step 4: Build COCO JSON
    print(f"\n[4/4] Building COCO OBB JSON...")
    coco_json = build_coco_json(filtered, categories, type_to_id, DATASET_PHASE)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"NASTaR_{DATASET_PHASE}_COCO_OBB_Detectron.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"Output saved to: {output_path}")
    print(f"{'=' * 60}")

    # If doing train/test splits, generate both
    if DATASET_PHASE in ("train", "test"):
        other_phase = "test" if DATASET_PHASE == "train" else "train"
        print(f"\n[Bonus] Also generating '{other_phase}' split...")
        other_filtered = filter_records(records, other_phase)
        other_json = build_coco_json(
            other_filtered, categories, type_to_id, other_phase
        )
        other_filename = f"NASTaR_{other_phase}_COCO_OBB_Detectron.json"
        other_path = os.path.join(OUTPUT_DIR, other_filename)
        with open(other_path, "w") as f:
            json.dump(other_json, f, indent=2)
        print(f"  Saved to: {other_path}")


if __name__ == "__main__":
    main()
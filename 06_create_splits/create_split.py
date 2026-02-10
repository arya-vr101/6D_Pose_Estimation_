import os
import shutil
import random
import json
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "04_dataset_generator" / "dataset_output"
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "mask"
LABELS_DIR = PROJECT_ROOT / "05_relabel_linemod" / "linemod_labels"
OUTPUT_DIR = CURRENT_DIR / "LINEMOD_mug"

#splits
def create_linemod_structure(images_dir,labels_dir,masks_dir,output_dir,train_ratio=0.7,test_ratio=0.2,val_ratio=0.1):
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")

    images = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))

    if not images:
        raise RuntimeError("No images found")

    print(f"🖼 Total images found: {len(images)}")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)

    splits = {"train": images[:train_end],"test": images[train_end:test_end], "val": images[test_end:]}

    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    for split, files in splits.items():
        print(f"Copying {split} set ({len(files)} files)")
        for img in files:
            name = img.stem

            shutil.copy2(img, output_dir / split / "images" / img.name)

            label = labels_dir / f"{name}.txt"
            if label.exists():
                shutil.copy2(label, output_dir / split / "labels" / label.name)

            mask = masks_dir / f"{name}.png"
            if mask.exists():
                shutil.copy2(mask, output_dir / split / "masks" / mask.name)

    for split, files in splits.items():
        with open(output_dir / f"{split}.txt", "w") as f:
            for img in files:
                f.write(f"{split}/images/{img.name}\n")

    with open(output_dir / "training_range.txt", "w") as f:
        f.write(f"0-{len(splits['train'])-1}")

    info = {"object": "mug","total_images": total,"splits": { "train": len(splits["train"]), "test": len(splits["test"]),
          "val": len(splits["val"])},"num_classes": 1,"class_names": ["mug"]}

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("\n LINEMOD dataset structure created successfully!")
    print(f"Output directory:\n{output_dir}")
# Main
if __name__ == "__main__":

    print("Project root :", PROJECT_ROOT)
    print("Images dir   :", IMAGES_DIR)
    print("Labels dir   :", LABELS_DIR)
    print("Masks dir    :", MASKS_DIR)

    for p in [IMAGES_DIR, LABELS_DIR, MASKS_DIR]:
        if not p.exists():
            raise RuntimeError(f"Missing directory:\n{p}")

    create_linemod_structure(images_dir=IMAGES_DIR,labels_dir=LABELS_DIR, masks_dir=MASKS_DIR,output_dir=OUTPUT_DIR)

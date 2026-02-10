import os
import cv2
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATASET_DIR = os.path.join( PROJECT_ROOT, "04_dataset_generator", "dataset_output")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "linemod_labels")

os.makedirs(OUTPUT_DIR, exist_ok=True)

#conversion
def convert_yolo_to_linemod(image_dir, label_dir, output_dir):

    if not os.path.exists(image_dir):
        raise RuntimeError(f"Images folder not found:\n{image_dir}")

    if not os.path.exists(label_dir):
        raise RuntimeError(f"Labels folder not found:\n{label_dir}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        raise RuntimeError("No images found in images directory")

    print(f"Images found: {len(image_files)}")

    valid_pairs = 0

    for img_file in image_files:
        name = os.path.splitext(img_file)[0]
        label_file = name + ".txt"

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_file}")
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        output_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            output_lines.append(f"{int(cls)} {x1} {y1} {x2} {y2}")

        if not output_lines:
            continue

        out_path = os.path.join(output_dir, label_file)
        with open(out_path, "w") as f:
            f.write("\n".join(output_lines))

        valid_pairs += 1

    print(f" Valid label-image pairs: {valid_pairs}")
    print("\n LINEMOD relabeling completed successfully!")
    print(f"Output saved to:\n{output_dir}")

if __name__ == "__main__":

    print("Dataset directory:", DATASET_DIR)
    print("Images directory :", IMAGE_DIR)
    print("Labels directory :", LABEL_DIR)

    convert_yolo_to_linemod(IMAGE_DIR,LABEL_DIR,OUTPUT_DIR)

    print("\nSample LINEMOD labels")
    print("Format: class x_min y_min x_max y_max")
    print("-" * 60)

    samples = sorted(os.listdir(OUTPUT_DIR))[:5]
    for s in samples:
        print(f"\n{s}")
        with open(os.path.join(OUTPUT_DIR, s)) as f:
            print(f.read())

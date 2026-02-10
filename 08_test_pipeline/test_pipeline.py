import json
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
class PipelineVerifier:
    def __init__(self):
        self.passed = []
        self.failed = []
    def header(self, title):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    def check_file(self, path, name):
        if path.exists():
            print(f"{name}")
            self.passed.append(name)
            return True
        else:
            print(f"{name}  →  {path}")
            self.failed.append(name)
            return False

    def check_dir(self, path, name, min_files=1):
        if not path.exists():
            print(f"{name} (missing)")
            self.failed.append(name)
            return False

        count = len(list(path.iterdir()))
        if count >= min_files:
            print(f"{name} ({count} files)")
            self.passed.append(name)
            return True
        else:
            print(f"{name} (empty)")
            self.failed.append(name)
            return False
    def phase_1(self):  #PHASE1
        self.header("PHASE 1: ChArUco Board")
        self.check_file(
            BASE_DIR / "01_generate_charuco" / "charuco_board.png",
            "ChArUco board image"
        )
    def phase_2(self): #PHASE2
        self.header("PHASE 2: Camera Calibration")

        calib_dir = BASE_DIR / "02_calibrate"
        self.check_file(calib_dir / "calibration_video.mp4", "Calibration video")

        params = calib_dir / "camera_params.json"
        if self.check_file(params, "Camera parameters"):
            with open(params) as f:
                p = json.load(f)
            print(f"  Image size: {p['image_width']} x {p['image_height']}")
    def phase_3(self):
        self.header("PHASE 3: 3D Model")

        model_path = BASE_DIR / "04_dataset_generator" / "mug.ply"
        if self.check_file(model_path, "Mug PLY model"):
            sys.path.append(str(BASE_DIR / "04_dataset_generator"))
            from pose_utils import load_ply_model
            verts = load_ply_model(str(model_path))
            print(f"  Vertices count: {len(verts)}")
    def phase_4(self):
        self.header("PHASE 4: Dataset Generation")

        d = BASE_DIR / "04_dataset_generator"
        out = d / "dataset_output"

        self.check_file(d / "mug.mp4", "Object video")
        self.check_dir(out / "images", "Images", 10)
        self.check_dir(out / "labels", "Labels", 10)
        self.check_dir(out / "mask", "Masks", 10)
        self.check_dir(out / "projections", "Projections", 10)
        self.check_file(out / "dataset_info.json", "Dataset info")
    def phase_5(self):
        self.header("PHASE 5: LINEMOD Dataset")

        base = BASE_DIR / "06_create_splits" / "LINEMOD_mug"

        self.check_dir(base / "train" / "images", "Train images")
        self.check_dir(base / "train" / "labels", "Train labels")
        self.check_dir(base / "train" / "masks", "Train masks")
        self.check_dir(base / "test" / "images", "Test images")
        self.check_dir(base / "val" / "images", "Validation images")

        for f in ["train.txt", "test.txt", "val.txt", "training_range.txt"]:
            self.check_file(base / f, f)
    def phase_6(self):
        self.header("PHASE 6: YOLO Training")

        self.check_file(
            BASE_DIR / "07_train_yolo" / "mug.yaml",
            "YOLO dataset config"
        )

        runs = BASE_DIR / "runs" / "detect" / "runs" / "train"
        if runs.exists():
            for r in runs.iterdir():
                print(f"Training run found: {r.name}")
                self.check_file(r / "weights" / "best.pt", "Best model")
        else:
            print("No training runs found")
    def run(self):
        self.phase_1()
        self.phase_2()
        self.phase_3()
        self.phase_4()
        self.phase_5()
        self.phase_6()

        self.header("SUMMARY")
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")

        if self.failed:
            print("Some steps are missing")
        else:
            print("PIPELINE VERIFIED SUCCESSFULLY")

if __name__ == "__main__":
    PipelineVerifier().run()

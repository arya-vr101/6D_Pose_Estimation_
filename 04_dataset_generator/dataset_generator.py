import cv2 # type: ignore
import numpy as np # type: ignore
import json
from pathlib import Path
import sys
from pathlib import Path

# Add project root to sys.path for pose_utils
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from pose_utils import (
    load_ply_model,
    compute_bounding_box_3d,
    project_points,
    draw_axis,
    draw_bounding_box_3d,
    draw_bounding_box_3d_wireframe,
    draw_orientation_arrow,
    create_mask,
    apply_offset
) # type: ignore

# ChArUco board parameters
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.02
DICTIONARY = cv2.aruco.DICT_6X6_250

# Mug offsets 
X_OFFSET = -0.15
Y_OFFSET = 0.01
Z_OFFSET = -0.18   
ROTATION_OFFSET = 0.0


class MugDatasetGenerator:
    """Generate dataset for mug 6D pose estimation"""

    def __init__(self, video_path, ply_path, camera_params_path, output_dir):

        self.video_path = video_path
        self.ply_path = ply_path
        self.output_dir = Path(output_dir)

        # Output folders
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.masks_dir = self.output_dir / "mask"
        self.projections_dir = self.output_dir / "projections"

        for d in [self.images_dir, self.labels_dir, self.masks_dir, self.projections_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load camera parameters
        with open(camera_params_path, "r") as f:
            params = json.load(f)

        self.camera_matrix = np.array(params["camera_matrix"], dtype=np.float32)
        self.dist_coeffs = np.array(params["distortion_coefficients"], dtype=np.float32)

        # Load 3D model
        print(f"Loading 3D model: {ply_path}")
        self.model_vertices = load_ply_model(ply_path).astype(np.float32)
        self.bbox_3d = compute_bounding_box_3d(self.model_vertices)

        # ArUco / ChArUco
        aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
        self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),SQUARE_LENGTH,MARKER_LENGTH,aruco_dict)

        # Initialize modern ChArUco Detector (OpenCV 4.8+)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)

    # Detect board pose
    def detect_board_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Modern detection call
        ch_corners, ch_ids, markers, ids = self.charuco_detector.detectBoard(gray)
        
        ch_corners_count: int = 0
        if ch_corners is not None and hasattr(ch_corners, '__len__'):
            ch_corners_count = len(ch_corners)
            
        if ch_corners_count < 4:
            return None, None

        # Estimate pose using solvePnP
        obj_points, img_points = self.board.matchImagePoints(ch_corners, ch_ids)
        
        obj_points_count: int = 0
        if obj_points is not None and hasattr(obj_points, '__len__'):
            obj_points_count = len(obj_points)
            
        if obj_points_count < 4:
            return None, None
            
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
        
        if not ok:
            return None, None

        return rvec, tvec
    def process_frame(self, frame, frame_idx):

        rvec, tvec = self.detect_board_pose(frame)
        if rvec is None:
            return False

        # Apply mug offset
        tvec_offset = apply_offset(tvec, X_OFFSET, Y_OFFSET, Z_OFFSET)

        # Project 3D bounding box
        bbox_3d_2d = project_points(self.bbox_3d,rvec,tvec_offset,self.camera_matrix,self.dist_coeffs)
        
        if not np.all(np.isfinite(bbox_3d_2d)):
            return False

        # Project FULL model for mask
        projected_model = project_points(self.model_vertices,rvec,tvec_offset,self.camera_matrix,self.dist_coeffs)
        
        if not np.all(np.isfinite(projected_model)):
            return False

        h, w = frame.shape[:2]
        x_min, y_min = bbox_3d_2d.min(axis=0).astype(int)
        x_max, y_max = bbox_3d_2d.max(axis=0).astype(int)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Visualization (Matching high-fidelity dual-box style)
        vis = frame.copy()
        
        # 1. Green Box: "3D Position" (Plain wireframe, thickness 1)
        vis = draw_bounding_box_3d_wireframe(vis, bbox_3d_2d, color=(0, 200, 0), thickness=1)
        
        # 2. Blue Box: "6D Pose" (Stylized, thickness 1)
        vis = draw_bounding_box_3d(vis, bbox_3d_2d, thickness=1)
        
        # 3. Orientation Indicator
        vis = draw_orientation_arrow(vis, rvec, tvec_offset, self.camera_matrix, self.dist_coeffs, length=0.04)

        # 4. 2D Bounding Box (Green, thickness 1)
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        name = f"frame_{frame_idx:06d}"

        # Save image + projection
        cv2.imwrite(str(self.images_dir / f"{name}.jpg"), frame)
        cv2.imwrite(str(self.projections_dir / f"{name}.jpg"), vis)

        # YOLO label
        xc = ((x_min + x_max) / 2) / w
        yc = ((y_min + y_max) / 2) / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h

        with open(self.labels_dir / f"{name}.txt", "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # FIXED MASK
        mask = create_mask(frame.shape, projected_model)
        cv2.imwrite(str(self.masks_dir / f"{name}.png"), mask)

        return True

    # Generate dataset
    def generate_dataset(self, skip_frames=5):

        cap = cv2.VideoCapture(self.video_path)
        frame_idx: int = 0
        saved: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                if self.process_frame(frame, saved):
                    # Ensure saved is treated as int
                    current_saved: int = int(saved)
                    saved = current_saved + 1 # type: ignore
                    print(f"Processed: {saved}", end="\r")

            frame_idx += 1

        cap.release()
        print("\nDataset generation complete.")

if __name__ == "__main__":

    BASE = Path(__file__).resolve().parent

    VIDEO = BASE / "mug.mp4"
    MODEL = BASE / "mug.ply"
    CAM = BASE.parent / "02_calibrate" / "camera_params.json"
    OUT = BASE / "dataset_output"

    for p in [VIDEO, MODEL, CAM]:
        if not p.exists():
            print(f"Missing file: {p}")
            sys.exit(1)

    gen = MugDatasetGenerator(video_path=str(VIDEO),ply_path=str(MODEL),camera_params_path=str(CAM),output_dir=str(OUT))

    gen.generate_dataset(skip_frames=5)

    print("\n DATASET GENERATION FINISHED")

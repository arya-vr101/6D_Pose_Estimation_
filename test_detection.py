import cv2 # type: ignore
import numpy as np # type: ignore
import json
from pathlib import Path
import sys

# Add paths for pose_utils
sys.path.append(str(Path(__file__).parent / "08_test_pipeline"))
sys.path.append(str(Path(__file__).parent / "04_dataset_generator"))

from pose_utils import ( # type: ignore
    load_ply_model, 
    compute_bounding_box_3d, 
    project_points, 
    draw_axis, 
    draw_bounding_box_3d,
    draw_bounding_box_3d_wireframe,
    draw_orientation_arrow,
    apply_offset,
    rotation_matrix_to_euler_angles
)

# ChArUco board parameters (must match dataset generation)
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.02
DICTIONARY = cv2.aruco.DICT_6X6_250

# Mug offsets (from dataset_generator.py)
X_OFFSET = 0.15
Y_OFFSET = 0.20
Z_OFFSET = 0.10

print("="*70)
print("6D Pose Estimation using ChArUco Board Detection")
print("="*70)
print(f"OpenCV Version: {cv2.__version__}")

# Load camera calibration
calib_path = Path(__file__).parent / "02_calibrate" / "camera_params.json"
with open(calib_path) as f:
    calib = json.load(f)
    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["distortion_coefficients"][0], dtype=np.float32)
print(f"✓ Loaded camera calibration")

# Load 3D model
model_path = Path(__file__).parent / "04_dataset_generator" / "mug.ply"
vertices = load_ply_model(str(model_path))
bbox_3d = compute_bounding_box_3d(vertices)
print(f"✓ Loaded 3D model: {len(vertices)} vertices")

# Setup ArUco/ChArUco detector - OpenCV 4.13.0 API 
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# Use CharucoDetector for OpenCV 4.8+
charuco_detector = cv2.aruco.CharucoDetector(board)
print(f"✓ Initialized CharucoDetector ({SQUARES_X}x{SQUARES_Y})")

# Load test image
img_path = "test_image.png"
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not load image '{img_path}'")
    sys.exit(1)

# Ensure img is not None for linter
assert img is not None

h, w = img.shape[:2]
print(f"Loaded test image: {w}x{h}")

# Scale camera matrix if image size differs from calibration
calib_w = calib["image_width"]
calib_h = calib["image_height"]

if w != calib_w or h != calib_h:
    print(f"Image resolution ({w}x{h}) differs from calibration ({calib_w}x{calib_h})")
    scale_x = w / calib_w
    scale_y = h / calib_h
    
    print(f"  Scaling camera matrix by: x={scale_x:.3f}, y={scale_y:.3f}")
    
    camera_matrix[0, 0] *= scale_x  # fx
    camera_matrix[0, 2] *= scale_x  # cx
    camera_matrix[1, 1] *= scale_y  # fy
    camera_matrix[1, 2] *= scale_y  # cy


print("\n" + "-"*70)
print("Detecting ChArUco board...")
print("-"*70)

# Detect ChArUco board - OpenCV 4.13.0 API 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectBoard returns: (charucoCorners, charucoIds, markerCorners, markerIds)
ch_corners, ch_ids, corners, ids = charuco_detector.detectBoard(gray)

# Check if markers were detected
# Check if markers were detected
ids_count: int = 0
if ids is not None and hasattr(ids, '__len__'):
    ids_count = len(ids)
    
if ids_count < 4:
    print(f"Error: Insufficient ArUco markers detected ({ids_count}/4+ needed)")
    print("  Make sure the ChArUco board is visible in the image")
    sys.exit(1)

print(f"✓ Detected {ids_count} ArUco markers")

# Check if ChArUco corners were detected
# Check if ChArUco corners were detected
ch_corners_count: int = 0
if ch_corners is not None and hasattr(ch_corners, '__len__'):
    ch_corners_count = len(ch_corners)
    
if ch_corners_count < 4:
    print(f"Error: Failed to detect ChArUco corners (got {ch_corners_count}/4+ needed)")
    print("  Try adjusting the board position or lighting")
    sys.exit(1)

print(f"✓ Detected {ch_corners_count} ChArUco corners")

# Estimate board pose - OpenCV 4.13.0 API 
# In OpenCV 4.8+, use board.matchImagePoints() instead of estimatePoseCharucoBoard()
try:
    # Try new API (OpenCV 4.8+)
    obj_points, img_points = board.matchImagePoints(ch_corners, ch_ids)
    
    obj_points_count: int = 0
    if obj_points is not None and hasattr(obj_points, '__len__'):
        obj_points_count = len(obj_points)
        
    if obj_points_count < 4:
        print(f"✗ Error: Failed to match image points (insufficient points: {obj_points_count})")
        sys.exit(1)
    
    # Use solvePnP to estimate pose
    ok, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
except AttributeError:
    # Fall back to old API (OpenCV < 4.8)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, camera_matrix, dist_coeffs, None, None
    )

if not ok or rvec is None or tvec is None:
    print(f"✗ Error: Failed to estimate board pose")
    sys.exit(1)

print(f"✓ Estimated board pose")

print("\n" + "-"*70)
print("Projecting 3D model...")
print("-"*70)

# Calculate Mug Pose relative to Board
# The mug is assumed to be sitting on the board.
# We need to transform the offset from BOARD frame to CAMERA frame.
# T_mug_cam = T_board_cam + R_board_cam * Offset_board

# Rotation matrix from vector
R_board, _ = cv2.Rodrigues(rvec)

# Mug offsets (Relative to board center-ish)
X_OFFSET = -0.01
Y_OFFSET = 0.14
Z_OFFSET = -0.018

# Offset in Board Frame (Meters)
# Offset in Board Frame (Meters)
# Based on Step 407, X=-0.01 and Y=0.14 worked best for alignment.
# We set Z to 0.0445 to center the origin relative to the mug's base and 12.5cm height
OFFSET_BOARD = np.array([-0.01, 0.14, 0.0445], dtype=np.float32)

# Apply rotation to offset
offset_cam = np.dot(R_board, OFFSET_BOARD)

# New tvec for mug (Centered)
tvec_mug = tvec.flatten() + offset_cam
tvec_mug = np.array(tvec_mug).reshape((3, 1))

# Centered Bounding Box (Meters)
# Adjusted to 12.5cm height to fit the mug physically in the frame
# Z goes from -0.0625 to 0.0625
bbox_3d = np.array([
    [-0.0425, -0.0425, -0.0625],
    [ 0.0425, -0.0425, -0.0625],
    [ 0.0425,  0.0425, -0.0625],
    [-0.0425,  0.0425, -0.0625],
    [-0.0425, -0.0425,  0.0625],
    [ 0.0425, -0.0425,  0.0625],
    [ 0.0425,  0.0425,  0.0625],
    [-0.0425,  0.0425,  0.0625]
], dtype=np.float32)

print(f"✓ Created tight symmetrical 3D bounding box (8.5x8.5x12.5 cm, centered)")

# Rotation for the mug (Just use board rotation since PLY is already Z-up)
rvec_mug_final = rvec
tvec_mug_final = tvec_mug

print(f"✓ Calculated mug pose (Board Z = Up)")

# Project 3D bounding box for 6D Pose (Purple, Exact)
bbox_3d_2d_6d = project_points(bbox_3d, rvec_mug_final, tvec_mug_final, camera_matrix, dist_coeffs)

# Project 3D bounding box for 3D Pose (Green, Wireframe)
# Scale only X and Y (1.1x) to make it wider but same height as the mug
bbox_3d_scaled = bbox_3d.copy()
bbox_3d_scaled[:, 0:2] *= 1.1 
bbox_3d_2d_3d = project_points(bbox_3d_scaled, rvec_mug_final, tvec_mug_final, camera_matrix, dist_coeffs)

print(f"✓ Projected 3D and 6D bounding boxes")

# Calculate 6D Pose (Translation and Rotation)
R_mug_final, _ = cv2.Rodrigues(rvec_mug_final)
euler_angles = rotation_matrix_to_euler_angles(R_mug_final)
# Convert to degrees for easier reading
euler_deg = np.degrees(euler_angles)

pose_text = [
    f"6D Pose Estimation:",
    f"X: {tvec_mug[0][0]:.3f} m",
    f"Y: {tvec_mug[1][0]:.3f} m",
    f"Z: {tvec_mug[2][0]:.3f} m",
    f"Roll:  {euler_deg[0]:.2f} deg",
    f"Pitch: {euler_deg[1]:.2f} deg",
    f"Yaw:   {euler_deg[2]:.2f} deg"
]

print("\nPose Information:")
for line in pose_text:
    print(f"  {line}")

# Create visualization (matching dual-pose request)
output_img = img.copy()

# 1. Draw 3D Position Box (Green wireframe)
output_img = draw_bounding_box_3d_wireframe(output_img, bbox_3d_2d_3d, color=(0, 255, 0), thickness=1)

# 2. Draw 6D Pose Bounding Box (Blue Edges + White corner dots)
# Blue in BGR is (255, 0, 0)
output_img = draw_bounding_box_3d(output_img, bbox_3d_2d_6d, pillar_color=(255, 0, 0), ring_color=(255, 0, 0), thickness=1)

# 3. Draw Orientation Indicator (Red pin/arrow)
output_img = draw_orientation_arrow(output_img, rvec_mug_final, tvec_mug_final, camera_matrix, dist_coeffs, length=0.06)

print(f"✓ Drew projection lines:")
print(f"  - 3D Pose Box (Green wireframe - 1.10x scale)")
print(f"  - 6D Pose Box (Blue edges + Dots)")
print(f"  - Orientation Indicator (Red pin pointing UP from center)")


# Save output
output_path = "test_output.png"
cv2.imwrite(output_path, output_img)
print(f"\n✓ Saved output to: {output_path}")

# Display comparison
print("\n" + "="*70)
print("Success! 6D Pose visualization complete.")
print("="*70)

print("\nDone!")
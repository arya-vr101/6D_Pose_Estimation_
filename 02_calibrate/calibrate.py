import cv2
import numpy as np
import json
import os

# ChArUco board parameters
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.04   # meters
MARKER_LENGTH = 0.02   # meters
DICTIONARY = cv2.aruco.DICT_6X6_250

# SPEED CONTROL (IMPORTANT)
SKIP_FRAMES = 5        # Process every 5th frame
MAX_GOOD_FRAMES = 60   # Stop after collecting enough frames


def calibrate_camera(
    video_path="02_calibrate/calibration_video.mp4",
    output_json="02_calibrate/camera_params.json"):
    
    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),SQUARE_LENGTH,MARKER_LENGTH,aruco_dict)

    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    print("Processing calibration video (FAST MODE)...")

    frame_count = 0
    used_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to speed up
        if frame_count % SKIP_FRAMES != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is not None:
            if len(marker_ids) > 0:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )

                if ret and charuco_corners is not None and len(charuco_corners) > 4:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    used_frames += 1

                    # Optional visualization
                    display = frame.copy()
                    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                    cv2.putText(display,f"Good frames: {used_frames}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
                    cv2.imshow("Calibration", display)
                    cv2.waitKey(1)

                    # Stop early when enough frames collected
                    if used_frames >= MAX_GOOD_FRAMES:
                        print("Enough good frames collected. Stopping early.")
                        break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed frames: {frame_count}")
    print(f"Used for calibration: {used_frames}")

    if used_frames < 15:
        print("Error: Not enough good frames for calibration!")
        return None

    if image_size is None:
        print("Error: Image size not determined!")
        return None

    # CALIBRATION
    print("\n Calibrating camera...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners,all_charuco_ids,board,image_size,None,None)

    if not ret:
        print(" Camera calibration failed!")
        return None

    if camera_matrix is None or dist_coeffs is None:
        print(" Camera calibration returned None matrices!")
        return None
    
    # Ensure they are numpy arrays for the linter
    camera_matrix = np.array(camera_matrix)
    dist_coeffs = np.array(dist_coeffs)

    # REPROJECTION ERROR
    assert camera_matrix is not None and dist_coeffs is not None
    total_error = 0
    for i in range(len(all_charuco_corners)):
        projected_points, _ = cv2.projectPoints(board.getChessboardCorners()[all_charuco_ids[i]],rvecs[i],tvecs[i],camera_matrix,dist_coeffs)
        error = cv2.norm(all_charuco_corners[i], projected_points, cv2.NORM_L2)
        total_error += error

    mean_error = total_error / len(all_charuco_corners)

    print("Calibration successful!")
    print(f"Reprojection error: {mean_error:.4f} pixels")


    # SAVE PARAMETERS
    if camera_matrix is not None and dist_coeffs is not None and image_size is not None:
        # Move to list format
        matrix_list = camera_matrix.tolist()
        dist_list = dist_coeffs.tolist()
        width = int(image_size[0])
        height = int(image_size[1])
        error_val = float(mean_error)
        
        camera_params = {
            "camera_matrix": matrix_list,
            "distortion_coefficients": dist_list,
            "image_width": width,
            "image_height": height,
            "reprojection_error": error_val,
            "frames_used": used_frames
        }
    else:
        print("Error: Calibration data missing!")
        return None

    with open(output_json, "w") as f:
        json.dump(camera_params, f, indent=4)

    print(f"\n Camera parameters saved to: {output_json}")
    print("\nCamera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)

    return camera_params


if __name__ == "__main__":
    video_path = "02_calibrate/calibration_video.mp4"

    if not os.path.exists(video_path):
        print(f" Error: Video file '{video_path}' not found!")
        print("Please record a calibration video and update the path.")
    else:
        calibrate_camera(video_path)

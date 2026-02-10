
import cv2
import numpy as np

# ChArUco board parameters
SQUARES_X = 5  # Number of squares in X direction
SQUARES_Y = 7  # Number of squares in Y direction
SQUARE_LENGTH = 0.04  # Square side length in meters (4cm)
MARKER_LENGTH = 0.02  # ArUco marker side length in meters (2cm)
DICTIONARY = cv2.aruco.DICT_6X6_250  # ArUco dictionary

def create_charuco_board():
    # Get predefined dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    
    # Create ChArUco board
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),SQUARE_LENGTH,MARKER_LENGTH,aruco_dict)
    
    # Generate board image 
    img_size_pixels = (2480, 3508) 
    
    board_img = board.generateImage(img_size_pixels)
    
    # Save the board
    output_path = "01_generate_charuco/charuco_board.png"
    cv2.imwrite(output_path, board_img)
    print(f"ChArUco board saved to: {output_path}")
    print(f"Board size: {SQUARES_X}x{SQUARES_Y}")
    print(f"Square length: {SQUARE_LENGTH}m")
    print(f"Marker length: {MARKER_LENGTH}m")
    print("\nPlease print this on A4 paper and use for calibration.")
    
    return board_img, board

if __name__ == "__main__":
    create_charuco_board()
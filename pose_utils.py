import numpy as np # type: ignore
import cv2 # type: ignore
from plyfile import PlyData # type: ignore

def load_ply_model(ply_path):
   
    plydata = PlyData.read(ply_path)
    vertices = np.vstack([plydata['vertex']['x'],
                          plydata['vertex']['y'],
                          plydata['vertex']['z']]).T
    return vertices.astype(np.float32)

def compute_bounding_box_3d(vertices):
    min_point = vertices.min(axis=0)
    max_point = vertices.max(axis=0)

    bbox_3d = np.array([[min_point[0], min_point[1], min_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [max_point[0], min_point[1], max_point[2]],
        [max_point[0], max_point[1], max_point[2]],
        [min_point[0], max_point[1], max_point[2]]
    ])
    return bbox_3d

def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length=0.08):
    """Draw 3D coordinate axes with RGB lines"""
    axis_points = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ])
    img_points = project_points(axis_points, rvec, tvec, camera_matrix, dist_coeffs).astype(int)
    origin = tuple(img_points[0])
    img = cv2.line(img, origin, tuple(img_points[1]), (0, 0, 255), 3)  # X - Red
    img = cv2.line(img, origin, tuple(img_points[2]), (0, 255, 0), 3)  # Y - Green
    img = cv2.line(img, origin, tuple(img_points[3]), (255, 0, 0), 3)  # Z - Blue
    
    # Circle at origin
    cv2.circle(img, origin, 4, (0, 0, 0), -1)
    cv2.circle(img, origin, 2, (255, 255, 255), -1)
    return img

def draw_orientation_arrow(img, rvec, tvec, camera_matrix, dist_coeffs, length=0.035):
    """Draw a specific red 'pin' orientation arrow (upward in object frame)"""
    axis_points = np.float32([[0, 0, 0], [0, 0, length]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2).astype(int)
    
    origin = (int(img_points[0][0]), int(img_points[0][1]))
    tip = (int(img_points[1][0]), int(img_points[1][1]))
    
    # Stem: Slightly thicker red line
    cv2.line(img, origin, tip, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Base: Solid black dot (matching reference)
    cv2.circle(img, origin, 3, (0, 0, 0), -1)
    
    # Head: Red dot with thin black outline
    cv2.circle(img, tip, 4, (0, 0, 0), -1) # Outline
    cv2.circle(img, tip, 3, (0, 0, 255), -1) # Red fill
    
    return img

def draw_bounding_box_2d(img, bbox_2d, color=(0, 255, 0), thickness=2):
    x_min, y_min, x_max, y_max = bbox_2d
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img

def draw_bounding_box_3d(img, bbox_3d_2d, pillar_color=(0, 255, 0), ring_color=(255, 0, 0), thickness=1):
    """Draw 3D bounding box with custom colors for vertical edges (pillars) and horizontal rings"""
    if not np.all(np.isfinite(bbox_3d_2d)):
        return img
        
    def get_pt(idx):
        return (int(round(float(bbox_3d_2d[idx][0]))), int(round(float(bbox_3d_2d[idx][1]))))

    try:
        # Draw pillars (Vertical edges)
        for i in range(4):
            cv2.line(img, get_pt(i), get_pt(i+4), pillar_color, int(thickness), cv2.LINE_AA)
            
        # Draw bottom ring
        for i in range(4):
            cv2.line(img, get_pt(i), get_pt((i+1)%4), ring_color, int(thickness), cv2.LINE_AA)
            
        # Draw top ring
        for i in range(4):
            cv2.line(img, get_pt(i+4), get_pt(((i+1)%4)+4), ring_color, int(thickness), cv2.LINE_AA)
            
        # Draw corner dots
        for i in range(len(bbox_3d_2d)):
            point = get_pt(i)
            cv2.circle(img, point, 4, (0, 0, 0), -1)    # Black outline
            cv2.circle(img, point, 2, (255, 255, 255), -1) # White center
    except (ValueError, OverflowError, cv2.error):
        pass
            
    return img


def draw_bounding_box_3d_wireframe(img, bbox_3d_2d, color=(0, 255, 0), thickness=1):
    # Ensure points are finite
    if not np.all(np.isfinite(bbox_3d_2d)):
        return img

    def get_pt(idx):
        return (int(round(float(bbox_3d_2d[idx][0]))), int(round(float(bbox_3d_2d[idx][1]))))

    try:
        # Draw pillars (Vertical edges)
        for i in range(4):
            cv2.line(img, get_pt(i), get_pt(i+4), color, int(thickness), cv2.LINE_AA)
            
        # Draw bottom ring
        for i in range(4):
            cv2.line(img, get_pt(i), get_pt((i+1)%4), color, int(thickness), cv2.LINE_AA)
            
        # Draw top ring
        for i in range(4):
            cv2.line(img, get_pt(i+4), get_pt(((i+1)%4)+4), color, int(thickness), cv2.LINE_AA)
            
        # Draw green corner dots for the wireframe (matching reference)
        for i in range(len(bbox_3d_2d)):
            cv2.circle(img, get_pt(i), 2, (0, 200, 0), -1)
    except (ValueError, OverflowError, cv2.error):
        pass
        
    return img


def create_mask(image_shape, projected_points):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if projected_points is None:
        return mask
    
    if len(projected_points) < 3:
        return mask

    # Convert to int pixel coordinates
    pts = projected_points.astype(np.int32)

    # Keep only points inside image
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    pts = pts[valid]

    if len(pts) < 3:
        return mask

    # Convex hull → object silhouette
    hull = cv2.convexHull(pts)

    # Fill mask
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def apply_offset(tvec, x_offset, y_offset, z_offset):
    new_tvec = tvec.copy()
    new_tvec[0] += x_offset
    new_tvec[1] += y_offset
    new_tvec[2] += z_offset
    return new_tvec

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

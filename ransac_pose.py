import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor

# Import Meta SAM
from segment_anything import sam_model_registry, SamPredictor

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Updated BASE_DIR to use absolute realpath
BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv2/ForkliftScene3Dv2"
IMG_IDX = 0  # We will use frame 00000

RGB_PATH = os.path.join(BASE_DIR, f"rgb/rgb_{IMG_IDX:05d}.png")
DEPTH_PATH = os.path.join(BASE_DIR, f"depth/depth_{IMG_IDX:05d}.npy")
INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
FORKLIFT_VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
OBB_PATH = os.path.join(BASE_DIR, "rigid_body_obbs.csv")

# Weights should be present in the same directory as this script
SAM_WEIGHTS = "sam_vit_h_4b8939.pth"

# Create results directory for visualizations in the current folder
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_transform_matrix(pos, quat):
    """Converts position and quaternion (w, x, y, z) to a 4x4 transformation matrix."""
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = pos
    return mat

def project_3d_to_pixel(pts_world, T_world2cam, fx, fy, cx, cy):
    """Projects 3D world points back onto the 2D image plane."""
    pts_cam_body = (T_world2cam[:3, :3] @ pts_world.T).T + T_world2cam[:3, 3]
    x_cv = -pts_cam_body[:, 1]
    y_cv = -pts_cam_body[:, 2]
    z_cv = pts_cam_body[:, 0]
    
    u = (x_cv / z_cv) * fx + cx
    v = (y_cv / z_cv) * fy + cy
    return np.vstack((u, v)).T

def draw_3d_obb(img, center_x, center_y, yaw, hx, hy, height, T_world2cam, intrinsics, color=(0, 255, 0)):
    """Draws a 3D bounding box on the image."""
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, -s], [s, c]])
    
    corners_2d = np.array([[hx, hy], [hx, -hy],[-hx, -hy],[-hx, hy]])
    corners_2d = (R_yaw @ corners_2d.T).T
    corners_2d[:, 0] += center_x
    corners_2d[:, 1] += center_y
    
    bottom_corners = np.column_stack((corners_2d, np.zeros(4)))
    top_corners = np.column_stack((corners_2d, np.full(4, height)))
    corners_3d = np.vstack((bottom_corners, top_corners))
    
    pixels = project_3d_to_pixel(corners_3d, T_world2cam, fx, fy, cx, cy).astype(int)
    
    for i in range(4):
        p1, p2 = pixels[i], pixels[(i+1)%4]
        p3, p4 = pixels[i+4], pixels[((i+1)%4)+4]
        p5, p6 = pixels[i], pixels[i+4]
        cv2.line(img, tuple(p1), tuple(p2), color, 2)
        cv2.line(img, tuple(p3), tuple(p4), color, 2)
        cv2.line(img, tuple(p5), tuple(p6), color, 2)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    print("Loading data...")
    img = cv2.imread(RGB_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = np.load(DEPTH_PATH)
    
    with open(INTRINSICS_PATH, 'r') as f:
        intrinsics_data = json.load(f)
        fx, fy = intrinsics_data['intrinsic_matrix'][0][0], intrinsics_data['intrinsic_matrix'][1][1]
        cx, cy = intrinsics_data['intrinsic_matrix'][0][2], intrinsics_data['intrinsic_matrix'][1][2]
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    # Load Poses
    views_df = pd.read_csv(FORKLIFT_VIEWS_PATH)
    row = views_df.iloc[IMG_IDX]
    
    T_fork2world = get_transform_matrix(
        np.array([row['forklift_pos_x'], row['forklift_pos_y'], row['forklift_pos_z']]),
        np.array([row['forklift_quat_w'], row['forklift_quat_x'], row['forklift_quat_y'], row['forklift_quat_z']])
    )
    T_cam2fork = get_transform_matrix(
        np.array([row['cam_pos_x'], row['cam_pos_y'], row['cam_pos_z']]),
        np.array([row['cam_quat_w'], row['cam_quat_x'], row['cam_quat_y'], row['cam_quat_z']])
    )
    T_cam2world = T_fork2world @ T_cam2fork
    T_world2cam = np.linalg.inv(T_cam2world)

    # --- STEP A: SEGMENTATION ---
    print("Step 1: Segmenting Pallet with SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS)
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)
    
    point_coords = np.array([[700, 380]])
    point_labels = np.array([1])
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
    mask = masks[0]

    # Save Step A Visualization
    vis_step1 = img.copy()
    vis_step1[mask] =[255, 0, 0] # Color mask blue
    vis_step1 = cv2.addWeighted(img, 0.6, vis_step1, 0.4, 0)
    cv2.circle(vis_step1, tuple(point_coords[0]), 5, (0, 0, 255), -1) # Draw prompt point
    cv2.imwrite(os.path.join(RESULTS_DIR, "01_segmentation_mask.png"), vis_step1)

    # --- STEP B: PROJECT TO 3D ---
    print("Step 2: Projecting contour to 3D World Space...")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pts = np.vstack(contours).squeeze()
    u, v = contour_pts[:, 0], contour_pts[:, 1]
    z = depth[v, u]
    
    valid = (z > 0) & (z < 15.0)
    u, v, z = u[valid], v[valid], z[valid]

    x_cv = (u - cx) * z / fx
    y_cv = (v - cy) * z / fy
    z_cv = z
    
    x_body, y_body, z_body = z_cv, -x_cv, -y_cv
    pts_body = np.vstack((x_body, y_body, z_body, np.ones_like(x_body))).T
    pts_world = (T_cam2world @ pts_body.T).T[:, :3]

    # Save Step B Visualization
    plt.figure(figsize=(8,8))
    plt.scatter(pts_world[:, 0], pts_world[:, 1], s=5, c='blue', label='Pallet Contour Points')
    plt.xlabel('X (World Coordinates)')
    plt.ylabel('Y (World Coordinates)')
    plt.title('Step 2: 3D Projected Contours (Bird\'s Eye View)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "02_3d_contours.png"))
    plt.close()

    # --- STEP C: RANSAC FRONT EDGE ESTIMATION ---
    print("Step 3: Running RANSAC on the front edge...")
    height_mask = (pts_world[:, 2] > -0.05) & (pts_world[:, 2] < 0.25)
    pts_world = pts_world[height_mask]
    
    sorted_indices = np.argsort(pts_world[:, 0])
    num_front_pts = int(len(pts_world) * 0.25)
    front_pts = pts_world[sorted_indices[:num_front_pts]]
    
    X = front_pts[:, 0]
    Y = front_pts[:, 1]
    
    ransac = RANSACRegressor(min_samples=3, residual_threshold=0.05)
    ransac.fit(Y.reshape(-1, 1), X)
    
    m = ransac.estimator_.coef_[0]
    c = ransac.estimator_.intercept_
    
    nx, ny = 1, -m
    norm = np.sqrt(nx**2 + ny**2)
    nx, ny = nx/norm, ny/norm
    if nx < 0:
        nx, ny = -nx, -ny
        
    est_yaw = np.arctan2(ny, nx)
    
    mid_Y = np.median(Y[ransac.inlier_mask_])
    mid_X = m * mid_Y + c
    
    est_center_x = mid_X + nx * 0.6
    est_center_y = mid_Y + ny * 0.6

    # Save Step C Visualization
    plt.figure(figsize=(8,8))
    plt.scatter(pts_world[:, 0], pts_world[:, 1], s=2, c='lightgray', label='All Contour Points')
    inliers = ransac.inlier_mask_
    outliers = np.logical_not(inliers)
    plt.scatter(X[inliers], Y[inliers], s=15, c='green', label='RANSAC Front Edge (Inliers)')
    plt.scatter(X[outliers], Y[outliers], s=15, c='red', label='Outliers')
    
    # Draw RANSAC Line
    line_Y = np.array([Y.min() - 0.2, Y.max() + 0.2])
    line_X = m * line_Y + c
    plt.plot(line_X, line_Y, color='blue', linewidth=2, label='Fitted Line')
    
    # Draw Vector to center
    plt.arrow(mid_X, mid_Y, nx*0.6, ny*0.6, color='purple', head_width=0.05, label='Normal to Center')
    plt.scatter([est_center_x], [est_center_y], c='purple', marker='x', s=100, label='Est Center')
    
    plt.xlabel('X (World Coordinates)')
    plt.ylabel('Y (World Coordinates)')
    plt.title('Step 3: RANSAC Edge Fitting & Center Projection')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "03_ransac_fit.png"))
    plt.close()

    # --- STEP D: COMPARE WITH GROUND TRUTH ---
    print("Step 4: Evaluating Benchmarks & Saving Final Image...")
    obbs_df = pd.read_csv(OBB_PATH)
    distances = np.sqrt((obbs_df['cx'] - est_center_x)**2 + (obbs_df['cy'] - est_center_y)**2)
    gt_row = obbs_df.iloc[distances.argmin()]
    gt_cx, gt_cy, gt_yaw = gt_row['cx'], gt_row['cy'], gt_row['yaw']
    hx, hy = gt_row['hx'], gt_row['hy']
    
    err_x = abs(est_center_x - gt_cx)
    err_y = abs(est_center_y - gt_cy)
    err_yaw_deg = abs(np.degrees(est_yaw) - np.degrees(gt_yaw))

    # --- STEP E: FINAL VISUALIZATION ---
    vis_img = img.copy()
    draw_3d_obb(vis_img, gt_cx, gt_cy, gt_yaw, hx, hy, 0.15, T_world2cam, intrinsics, color=(0, 255, 0))
    draw_3d_obb(vis_img, est_center_x, est_center_y, est_yaw, hx, hy, 0.15, T_world2cam, intrinsics, color=(0, 0, 255))
    
    text =[
        f"GREEN: Ground Truth | RED: RANSAC Estimate",
        f"Error X: {err_x*100:.2f} cm",
        f"Error Y: {err_y*100:.2f} cm",
        f"Error Yaw: {err_yaw_deg:.2f} deg"
    ]
    for i, line in enumerate(text):
        cv2.putText(vis_img, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    cv2.imwrite(os.path.join(RESULTS_DIR, "04_final_benchmark.png"), vis_img)
    print(f"Success! Check the '{os.path.abspath(RESULTS_DIR)}' folder for the step-by-step images.")

if __name__ == "__main__":
    main()
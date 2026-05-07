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
BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv2/ForkliftScene3Dv2"
IMG_IDX = 0  

RGB_PATH = os.path.join(BASE_DIR, f"rgb/rgb_{IMG_IDX:05d}.png")
DEPTH_PATH = os.path.join(BASE_DIR, f"depth/depth_{IMG_IDX:05d}.npy")
INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
FORKLIFT_VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
OBB_PATH = os.path.join(BASE_DIR, "rigid_body_obbs.csv")

SAM_WEIGHTS = "sam_vit_h_4b8939.pth"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_transform_matrix(pos, quat):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = pos
    return mat

def project_3d_to_pixel(pts_world, T_world2cam, fx, fy, cx, cy):
    pts_cam_body = (T_world2cam[:3, :3] @ pts_world.T).T + T_world2cam[:3, 3]
    x_cv = -pts_cam_body[:, 1]
    y_cv = -pts_cam_body[:, 2]
    z_cv = pts_cam_body[:, 0]
    u = (x_cv / z_cv) * fx + cx
    v = (y_cv / z_cv) * fy + cy
    return np.vstack((u, v)).T

def draw_point_3d(img, pt_3d, T_world2cam, intrinsics, color, radius=8):
    """Draws a solid circle at a specific 3D world coordinate."""
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    pixels = project_3d_to_pixel(np.array([pt_3d]), T_world2cam, fx, fy, cx, cy)
    if len(pixels) > 0:
        cv2.circle(img, tuple(pixels[0].astype(int)), radius, color, -1)

def draw_3d_obb(img, center_x, center_y, yaw, hx, hy, height, T_world2cam, intrinsics, color):
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
        cv2.line(img, tuple(pixels[i]), tuple(pixels[(i+1)%4]), color, 2)
        cv2.line(img, tuple(pixels[i+4]), tuple(pixels[((i+1)%4)+4]), color, 2)
        cv2.line(img, tuple(pixels[i]), tuple(pixels[i+4]), color, 2)

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

    # Load Transforms
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
    
    point_coords = np.array([[446, 396]]) # Ensure this points to a pallet
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=np.array([1]), multimask_output=False)
    mask = masks[0]

    # Visualization A: Segmentation Mask
    vis_step1 = img.copy()
    vis_step1[mask] =[255, 0, 0] # Color mask blue
    vis_step1 = cv2.addWeighted(img, 0.6, vis_step1, 0.4, 0)
    cv2.circle(vis_step1, tuple(point_coords[0]), 5, (0, 0, 255), -1) # Draw prompt point
    cv2.imwrite(os.path.join(RESULTS_DIR, "01_segmentation_mask.png"), vis_step1)

    # --- STEP B: EXTRACT FRONT EDGE EXCLUSIVELY ---
    print("Step 2: Extracting Front Edge from mask...")
    v_coords, u_coords = np.where(mask)
    df_mask = pd.DataFrame({'u': u_coords, 'v': v_coords})
    
    # For every column (u), find the maximum row (v), which is the bottom edge
    bottom_edge = df_mask.groupby('u')['v'].max().reset_index()
    u, v = bottom_edge['u'].values, bottom_edge['v'].values
    z = depth[v, u]
    
    valid = (z > 0) & (z < 15.0)
    u, v, z = u[valid], v[valid], z[valid]

    # Project to 3D World Space
    x_cv, y_cv = (u - cx) * z / fx, (v - cy) * z / fy
    pts_body = np.vstack((z, -x_cv, -y_cv, np.ones_like(z))).T
    front_pts_world = (T_cam2world @ pts_body.T).T[:, :3]

    # Visualization B: 3D Front Edge
    plt.figure(figsize=(8,8))
    plt.scatter(front_pts_world[:, 0], front_pts_world[:, 1], s=10, c='blue', label='Front Edge Points')
    plt.xlabel('X (World Coordinates)')
    plt.ylabel('Y (World Coordinates)')
    plt.title("Step 2: 3D Projected Front Edge (Bird's Eye View)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "02_3d_front_edge.png"))
    plt.close()

    # --- STEP C: RANSAC FRONT CENTER ESTIMATION ---
    print("Step 3: Fitting RANSAC to the front face...")
    X = front_pts_world[:, 0]
    Y = front_pts_world[:, 1]
    
    ransac = RANSACRegressor(min_samples=3, residual_threshold=0.03) # Tighter threshold
    ransac.fit(Y.reshape(-1, 1), X)
    
    m = ransac.estimator_.coef_[0]
    c = ransac.estimator_.intercept_
    
    inlier_mask = ransac.inlier_mask_
    inlier_Y = Y[inlier_mask]
    
    # CALCULATE FRONT CENTER
    min_Y, max_Y = np.min(inlier_Y), np.max(inlier_Y)
    est_front_y = (min_Y + max_Y) / 2.0
    est_front_x = m * est_front_y + c 
    
    # Calculate Yaw
    nx, ny = -1, m
    norm = np.sqrt(nx**2 + ny**2)
    nx, ny = nx/norm, ny/norm
    est_yaw = np.arctan2(-ny, -nx)

    # Visualization C: RANSAC Line Fit
    plt.figure(figsize=(8,8))
    outliers = np.logical_not(inlier_mask)
    plt.scatter(X[inlier_mask], Y[inlier_mask], s=15, c='green', label='RANSAC Inliers')
    plt.scatter(X[outliers], Y[outliers], s=15, c='red', label='Outliers')
    
    line_Y = np.array([Y.min() - 0.1, Y.max() + 0.1])
    line_X = m * line_Y + c
    plt.plot(line_X, line_Y, color='blue', linewidth=2, label='Fitted Line')
    plt.scatter([est_front_x], [est_front_y], c='purple', marker='X', s=150, label='Estimated Front Center')
    
    plt.xlabel('X (World Coordinates)')
    plt.ylabel('Y (World Coordinates)')
    plt.title("Step 3: RANSAC Edge Fitting & Center Prediction")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "03_ransac_fit.png"))
    plt.close()

    # --- STEP D: COMPARE WITH GROUND TRUTH ---
    print("Step 4: Evaluating Front-Center Benchmarks...")
    obbs_df = pd.read_csv(OBB_PATH)
    distances = np.sqrt((obbs_df['cx'] - est_front_x)**2 + (obbs_df['cy'] - est_front_y)**2)
    gt_row = obbs_df.iloc[distances.argmin()]
    gt_cx, gt_cy, gt_yaw = gt_row['cx'], gt_row['cy'], gt_row['yaw']
    hx, hy = gt_row['hx'], gt_row['hy']
    
    c_cos, c_sin = np.cos(gt_yaw), np.sin(gt_yaw)
    R_mat = np.array([[c_cos, -c_sin], [c_sin, c_cos]])
    corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy],[-hx, hy]])
    corners_world = (R_mat @ corners.T).T + np.array([gt_cx, gt_cy])
    
    sorted_corners = corners_world[np.argsort(corners_world[:, 0])]
    gt_front_x = (sorted_corners[0][0] + sorted_corners[1][0]) / 2.0
    gt_front_y = (sorted_corners[0][1] + sorted_corners[1][1]) / 2.0

    err_x = abs(est_front_x - gt_front_x)
    err_y = abs(est_front_y - gt_front_y)
    err_yaw_deg = abs(np.degrees(est_yaw) - np.degrees(gt_yaw))

    # --- STEP E: FINAL VISUALIZATION ---
    vis_img = img.copy()
    
    draw_3d_obb(vis_img, gt_cx, gt_cy, gt_yaw, hx, hy, 0.15, T_world2cam, intrinsics, color=(0, 255, 0))
    
    draw_point_3d(vis_img,[gt_front_x, gt_front_y, 0.05], T_world2cam, intrinsics, color=(0, 255, 0), radius=10)
    draw_point_3d(vis_img, [est_front_x, est_front_y, 0.05], T_world2cam, intrinsics, color=(0, 0, 255), radius=10)
    
    text =[
        f"TARGET: Front Center of Pallet",
        f"GREEN Dot: Ground Truth | RED Dot: RANSAC Est",
        f"Front Center X Error: {err_x*100:.2f} cm",
        f"Front Center Y Error: {err_y*100:.2f} cm",
        f"Yaw Error: {err_yaw_deg:.2f} deg"
    ]
    for i, line in enumerate(text):
        cv2.putText(vis_img, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    cv2.imwrite(os.path.join(RESULTS_DIR, "04_front_center_benchmark.png"), vis_img)
    print(f"Success! Check the '{os.path.abspath(RESULTS_DIR)}' folder for the 4 step-by-step images.")

if __name__ == "__main__":
    main()
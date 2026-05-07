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
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    pixels = project_3d_to_pixel(np.array([pt_3d]), T_world2cam, fx, fy, cx, cy)
    if len(pixels) > 0:
        cv2.circle(img, tuple(pixels[0].astype(int)), radius, color, -1)

def draw_3d_obb(img, center_x, center_y, yaw, hx, hy, height, T_world2cam, intrinsics, color):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, -s],[s, c]])
    
    corners_2d = np.array([[hx, hy],[hx, -hy],[-hx, -hy],[-hx, hy]])
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

def draw_pose_axes(img, x, y, z, yaw, T_world2cam, intrinsics, length=0.4, thickness=3):
    """Draws 3D RGB axes representing the 6D Pose."""
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    
    origin = np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    
    # Calculate Unit Vectors for X, Y, Z axes
    pt_x = origin + np.array([c, s, 0]) * length
    pt_y = origin + np.array([-s, c, 0]) * length
    pt_z = origin + np.array([0, 0, length])
    
    # Project all 4 points to 2D
    pts_3d = np.array([origin, pt_x, pt_y, pt_z])
    pixels = project_3d_to_pixel(pts_3d, T_world2cam, fx, fy, cx, cy)
    
    if len(pixels) == 4:
        p_org = tuple(pixels[0].astype(int))
        p_x = tuple(pixels[1].astype(int))
        p_y = tuple(pixels[2].astype(int))
        p_z = tuple(pixels[3].astype(int))
        
        # Draw Arrows (OpenCV colors are BGR)
        cv2.arrowedLine(img, p_org, p_x, (0, 0, 255), thickness, tipLength=0.2) # X Axis = Red
        cv2.arrowedLine(img, p_org, p_y, (0, 255, 0), thickness, tipLength=0.2) # Y Axis = Green
        cv2.arrowedLine(img, p_org, p_z, (255, 0, 0), thickness, tipLength=0.2) # Z Axis = Blue

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    print(f"Loading data for Image Index: {IMG_IDX:05d}...")
    img = cv2.imread(RGB_PATH)
    img_h, img_w = img.shape[:2]
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

    print("Loading SAM Model...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS)
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)
    
    # Pre-load Ground Truth Obbs and Filter to ONLY include Pallets
    obbs_df = pd.read_csv(OBB_PATH)
    pallets_only_df = obbs_df[obbs_df['prim_path'].str.contains('Pallet', case=False, na=False)].copy()

    pallet_counter = 1

    for index, gt_row in pallets_only_df.iterrows():
        gt_cx, gt_cy, gt_yaw = gt_row['cx'], gt_row['cy'], gt_row['yaw']
        hx, hy = gt_row['hx'], gt_row['hy']
        
        c_cos, c_sin = np.cos(gt_yaw), np.sin(gt_yaw)
        R_mat = np.array([[c_cos, -c_sin],[c_sin, c_cos]])
        corners = np.array([[hx, hy],[hx, -hy],[-hx, -hy],[-hx, hy]])
        corners_world = (R_mat @ corners.T).T + np.array([gt_cx, gt_cy])
        
        sorted_corners = corners_world[np.argsort(corners_world[:, 0])]
        gt_front_x = (sorted_corners[0][0] + sorted_corners[1][0]) / 2.0
        gt_front_y = (sorted_corners[0][1] + sorted_corners[1][1]) / 2.0

        pt_3d = np.array([[gt_front_x, gt_front_y, 0.075]])
        
        pts_cam_body = (T_world2cam[:3, :3] @ pt_3d.T).T + T_world2cam[:3, 3]
        z_cv = pts_cam_body[0, 0] 
        if z_cv <= 0:
            continue 
            
        pixel = project_3d_to_pixel(pt_3d, T_world2cam, fx, fy, cx, cy)[0]
        px, py = int(pixel[0]), int(pixel[1])

        if not (0 <= px < img_w and 0 <= py < img_h):
            continue 

        print(f"\n==========================================")
        print(f"Processing Visible Pallet {pallet_counter} | Prompting SAM at: ({px}, {py})")
        
        img_name_str = f"{IMG_IDX:05d}"
        pallet_dir = os.path.join(RESULTS_DIR, img_name_str, str(pallet_counter))
        os.makedirs(pallet_dir, exist_ok=True)

        # --- STEP A: SEGMENTATION ---
        point_coords = np.array([[px, py]])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=np.array([1]), multimask_output=False)
        mask = masks[0]

        vis_step1 = img.copy()
        vis_step1[mask] =[255, 0, 0] 
        vis_step1 = cv2.addWeighted(img, 0.6, vis_step1, 0.4, 0)
        cv2.circle(vis_step1, tuple(point_coords[0]), 5, (0, 0, 255), -1) 
        cv2.imwrite(os.path.join(pallet_dir, "01_segmentation_mask.png"), vis_step1)

        # --- STEP B: EXTRACT FRONT EDGE EXCLUSIVELY ---
        v_coords, u_coords = np.where(mask)
        if len(u_coords) == 0:
            print("SAM failed to find a mask here! Skipping...")
            continue
            
        df_mask = pd.DataFrame({'u': u_coords, 'v': v_coords})
        bottom_edge = df_mask.groupby('u')['v'].max().reset_index()
        u, v = bottom_edge['u'].values, bottom_edge['v'].values
        z = depth[v, u]
        
        valid = (z > 0) & (z < 15.0)
        u, v, z = u[valid], v[valid], z[valid]

        if len(z) < 3:
            print("Not enough valid depth points on the edge! Skipping...")
            continue

        x_cv, y_cv = (u - cx) * z / fx, (v - cy) * z / fy
        pts_body = np.vstack((z, -x_cv, -y_cv, np.ones_like(z))).T
        front_pts_world = (T_cam2world @ pts_body.T).T[:, :3]

        # --- STEP C: DEPTH TRUNCATION & RANSAC ---
        min_x = np.min(front_pts_world[:, 0])
        depth_mask = front_pts_world[:, 0] < (min_x + 0.20)
        front_pts_world = front_pts_world[depth_mask]

        X = front_pts_world[:, 0]
        Y = front_pts_world[:, 1]
        
        ransac = RANSACRegressor(min_samples=3, residual_threshold=0.03) 
        ransac.fit(Y.reshape(-1, 1), X)
        
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_
        
        inlier_mask = ransac.inlier_mask_
        inlier_Y = Y[inlier_mask]
        
        min_Y, max_Y = np.min(inlier_Y), np.max(inlier_Y)
        est_front_y = (min_Y + max_Y) / 2.0
        est_front_x = m * est_front_y + c 
        
        nx, ny = -1, m
        norm = np.sqrt(nx**2 + ny**2)
        nx, ny = nx/norm, ny/norm
        est_yaw = np.arctan2(-ny, -nx)

        # Re-calculate 6D Geometric Center of the pallet from the Front Center
        est_cx = est_front_x + np.cos(est_yaw) * hx
        est_cy = est_front_y + np.sin(est_yaw) * hx

        # Visualizations B & C
        plt.figure(figsize=(8,8))
        plt.scatter(front_pts_world[:, 0], front_pts_world[:, 1], s=10, c='blue', label='Front Edge Points (Truncated)')
        plt.xlabel('X (World Coordinates)')
        plt.ylabel('Y (World Coordinates)')
        plt.title(f"Pallet {pallet_counter} - 3D Projected Front Edge")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(pallet_dir, "02_3d_front_edge.png"))
        plt.close()

        plt.figure(figsize=(8,8))
        outliers = np.logical_not(inlier_mask)
        plt.scatter(X[inlier_mask], Y[inlier_mask], s=15, c='green', label='RANSAC Inliers')
        plt.scatter(X[outliers], Y[outliers], s=15, c='red', label='Outliers')
        
        line_Y = np.array([Y.min() - 0.1, Y.max() + 0.1])
        line_X = m * line_Y + c
        plt.plot(line_X, line_Y, color='blue', linewidth=2, label='Fitted Line')
        plt.scatter([est_front_x],[est_front_y], c='purple', marker='X', s=150, label='Estimated Front Center')
        
        plt.xlabel('X (World Coordinates)')
        plt.ylabel('Y (World Coordinates)')
        plt.title(f"Pallet {pallet_counter} - RANSAC Edge Fitting")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(pallet_dir, "03_ransac_fit.png"))
        plt.close()

        # --- STEP D: COMPARE WITH GROUND TRUTH ---
        err_x = abs(est_front_x - gt_front_x)
        err_y = abs(est_front_y - gt_front_y)
        err_yaw_deg = abs(np.degrees(est_yaw) - np.degrees(gt_yaw))

        text_block =[
            f"PALLET {pallet_counter}",
            f"Front Center X Error: {err_x*100:.2f} cm",
            f"Front Center Y Error: {err_y*100:.2f} cm",
            f"Yaw Error: {err_yaw_deg:.2f} deg"
        ]

        # --- STEP E: FRONT CENTER VISUALIZATION ---
        vis_img = img.copy()  
        draw_3d_obb(vis_img, gt_cx, gt_cy, gt_yaw, hx, hy, 0.15, T_world2cam, intrinsics, color=(0, 255, 0))
        draw_point_3d(vis_img,[gt_front_x, gt_front_y, 0.05], T_world2cam, intrinsics, color=(0, 255, 0), radius=10)
        draw_point_3d(vis_img,[est_front_x, est_front_y, 0.05], T_world2cam, intrinsics, color=(0, 0, 255), radius=10)
        
        for i, line in enumerate(text_block + ["GREEN Dot: GT | RED Dot: RANSAC"]):
            cv2.putText(vis_img, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.imwrite(os.path.join(pallet_dir, "04_front_center_benchmark.png"), vis_img)

        # --- STEP F: FULL 6D POSE (AXES) VISUALIZATION ---
        vis_img_6d = img.copy()
        
        # Ground Truth Pose (Thick Arrows)
        draw_pose_axes(vis_img_6d, gt_cx, gt_cy, 0.15, gt_yaw, T_world2cam, intrinsics, thickness=6, length=0.4)
        
        # Estimated Pose (Thin Arrows)
        draw_pose_axes(vis_img_6d, est_cx, est_cy, 0.15, est_yaw, T_world2cam, intrinsics, thickness=2, length=0.4)
        
        for i, line in enumerate(text_block +["THICK Axes: GT | THIN Axes: RANSAC Est"]):
            cv2.putText(vis_img_6d, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(pallet_dir, "05_6d_pose_benchmark.png"), vis_img_6d)

        print(f"Success! Results saved to: {pallet_dir}")
        pallet_counter += 1

    print(f"\nAll visible pallets processed successfully!")

if __name__ == "__main__":
    main()
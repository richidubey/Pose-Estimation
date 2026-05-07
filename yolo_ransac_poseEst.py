import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from ultralytics import YOLO

# 1. SETUP & CONFIGURATION
BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv3/"
YOLO_WEIGHTS_PATH = "/nethome/rdubey36/poseEst/large_100_v1/best.pt"

IMAGE_IDS = ['00000', '00001', '00003', '00004', '00005', '00010', '00020', '00025']

INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
FORKLIFT_VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
POSES_PATH = os.path.join(BASE_DIR, "rigid_body_poses.csv")
RESULTS_DIR = "results_yolo_poses"

# Pallet Dimension Mapping (Half-extents in meters)
# Wood pallets are typically 1.2x0.8, Blue/Black rackable are often 1.2x1.0
PALLET_DIMS = {
    'BlockPallet': {'hx': 0.6, 'hy': 0.4, 'hz': 0.075},
    'RackablePallet': {'hx': 0.6, 'hy': 0.5, 'hz': 0.075},
    'DEFAULT': {'hx': 0.6, 'hy': 0.4, 'hz': 0.075}
}

# 2. HELPER FUNCTIONS

def get_transform_matrix(pos, quat):
    # Scipy expects [x, y, z, w], CSV has [w, x, y, z]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = pos
    return mat

def get_yaw_from_quat(quat):
    # Returns yaw in radians around Z axis
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = r.as_euler('zyx')
    return euler[0]

def project_3d_to_pixel(pts_world, T_world2cam, fx, fy, cx, cy):
    pts_cam_body = (T_world2cam[:3, :3] @ pts_world.T).T + T_world2cam[:3, 3]
    x_cv, y_cv, z_cv = -pts_cam_body[:, 1], -pts_cam_body[:, 2], pts_cam_body[:, 0]
    u, v = (x_cv / z_cv) * fx + cx, (y_cv / z_cv) * fy + cy
    return np.vstack((u, v)).T

def draw_pose_axes(img, x, y, z, yaw, T_world2cam, intrinsics, length=0.4, thickness=3):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    origin = np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    pts_3d = np.array([origin, origin + [c*length, s*length, 0], origin + [-s*length, c*length, 0], origin + [0, 0, length]])
    pixels = project_3d_to_pixel(pts_3d, T_world2cam, fx, fy, cx, cy).astype(int)
    if len(pixels) == 4:
        cv2.arrowedLine(img, tuple(pixels[0]), tuple(pixels[1]), (0, 0, 255), thickness) # X-Red
        cv2.arrowedLine(img, tuple(pixels[0]), tuple(pixels[2]), (0, 255, 0), thickness) # Y-Green
        cv2.arrowedLine(img, tuple(pixels[0]), tuple(pixels[3]), (255, 0, 0), thickness) # Z-Blue

def calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy, hz=0.075):
    base_corners = np.array(list(itertools.product([hx, -hx], [hy, -hy], [hz, -hz])))
    def transform(corners, cx, cy, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        R_mat = np.array([[c, -s, 0],[s, c, 0], [0, 0, 1]])
        return (R_mat @ corners.T).T + np.array([cx, cy, 0.075]) 
    gt_corners = transform(base_corners, gt_cx, gt_cy, gt_yaw)
    est_c1 = transform(base_corners, est_cx, est_cy, est_yaw)
    est_c2 = transform(base_corners, est_cx, est_cy, est_yaw + np.pi)
    return min(np.linalg.norm(gt_corners - est_c1, axis=1).mean(), 
               np.linalg.norm(gt_corners - est_c2, axis=1).mean())

# 3. MAIN PIPELINE

def main():
    print("Loading Models and CSVs...")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    
    with open(INTRINSICS_PATH, 'r') as f:
        data = json.load(f)
        intrinsics = {'fx': data['intrinsic_matrix'][0][0], 'fy': data['intrinsic_matrix'][1][1], 
                      'cx': data['intrinsic_matrix'][0][2], 'cy': data['intrinsic_matrix'][1][2]}

    views_df = pd.read_csv(FORKLIFT_VIEWS_PATH)
    poses_df = pd.read_csv(POSES_PATH)
    
    global_results = []

    for img_id in IMAGE_IDS:
        print(f"Processing Image {img_id}...")
        rgb_path = os.path.join(BASE_DIR, f"rgb/rgb_{img_id}.png")
        depth_path = os.path.join(BASE_DIR, f"depth/depth_{img_id}.npy")
        if not os.path.exists(rgb_path): continue

        img = cv2.imread(rgb_path)
        depth = np.load(depth_path)
        
        # 1. Get Camera Transform
        v_row = views_df[views_df['rgb_path'].str.contains(img_id)].iloc[0]
        T_cam2world = get_transform_matrix(np.array([v_row['forklift_pos_x'], v_row['forklift_pos_y'], v_row['forklift_pos_z']]),
                                           np.array([v_row['forklift_quat_w'], v_row['forklift_quat_x'], v_row['forklift_quat_y'], v_row['forklift_quat_z']])) \
                      @ get_transform_matrix(np.array([v_row['cam_pos_x'], v_row['cam_pos_y'], v_row['cam_pos_z']]),
                                           np.array([v_row['cam_quat_w'], v_row['cam_quat_x'], v_row['cam_quat_y'], v_row['cam_quat_z']]))
        T_world2cam = np.linalg.inv(T_cam2world)

        # 2. Get All Pallet GTs for this image (Assuming row index matches image index)
        # Note: If your row order differs, you'll need to match Sample ID
        img_idx = int(img_id)
        p_row = poses_df.iloc[img_idx]
        
        frame_gt_pallets = []
        for i in range(5): # Parsing up to 5 pallets as per your CSV snippet
            name = str(p_row[f'pallet_{i}_name'])
            if 'Pallet' not in name: continue
            
            dim_key = 'RackablePallet' if 'Rackable' in name else 'BlockPallet'
            dims = PALLET_DIMS.get(dim_key, PALLET_DIMS['DEFAULT'])
            
            quat = [p_row[f'pallet_{i}_quat_w'], p_row[f'pallet_{i}_quat_x'], p_row[f'pallet_{i}_quat_y'], p_row[f'pallet_{i}_quat_z']]
            frame_gt_pallets.append({
                'cx': p_row[f'pallet_{i}_pos_x'], 'cy': p_row[f'pallet_{i}_pos_y'],
                'yaw': get_yaw_from_quat(quat), 'hx': dims['hx'], 'hy': dims['hy']
            })

        # 3. YOLO Detection
        results = yolo_model(img, conf=0.4, verbose=False)[0]
        if results.masks is None: continue

        masks = results.masks.data.cpu().numpy()
        img_dir = os.path.join(RESULTS_DIR, img_id)
        os.makedirs(img_dir, exist_ok=True)

        for m_idx, mask in enumerate(masks):
            if mask.shape[0] != img.shape[0]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Edge Extraction
            v_coords, u_coords = np.where(mask > 0.5)
            if len(u_coords) < 20: continue
            df_m = pd.DataFrame({'u': u_coords, 'v': v_coords})
            edge = df_m.groupby('u')['v'].max().reset_index()
            u, v = edge['u'].values, edge['v'].values
            z_vals = depth[v, u]
            
            # 3D Projection
            valid = (z_vals > 0.1) & (z_vals < 15.0)
            u, v, z_vals = u[valid], v[valid], z_vals[valid]
            x_cv, y_cv = (u - intrinsics['cx']) * z_vals / intrinsics['fx'], (v - intrinsics['cy']) * z_vals / intrinsics['fy']
            pts_body = np.vstack((z_vals, -x_cv, -y_cv, np.ones_like(z_vals))).T
            pts_world = (T_cam2world @ pts_body.T).T[:, :3]

            # RANSAC
            try:
                # Isolate front face points (close to camera)
                pts_trunc = pts_world[pts_world[:, 0] < (np.min(pts_world[:, 0]) + 0.15)]
                ransac = RANSACRegressor(residual_threshold=0.03).fit(pts_trunc[:, 1].reshape(-1,1), pts_trunc[:, 0])
                
                est_front_y = np.mean(pts_trunc[:, 1][ransac.inlier_mask_])
                est_front_x = ransac.predict([[est_front_y]])[0]
                m_slope = ransac.estimator_.coef_[0]
                est_yaw = np.arctan2(-m_slope, 1) # Simplified yaw from line slope
            except: continue

            # Match to closest GT pallet
            best_gt = min(frame_gt_pallets, key=lambda p: np.hypot(p['cx'] - est_front_x, p['cy'] - est_front_y))
            
            # Final Pose
            est_cx = est_front_x + np.cos(est_yaw) * best_gt['hx']
            est_cy = est_front_y + np.sin(est_yaw) * best_gt['hx']
            
            # Metrics
            add_s = calculate_adds(best_gt['cx'], best_gt['cy'], best_gt['yaw'], est_cx, est_cy, est_yaw, best_gt['hx'], best_gt['hy'])
            global_results.append({'img': img_id, 'add_s_cm': add_s*100})

            # Visualization
            vis = img.copy()
            draw_pose_axes(vis, best_gt['cx'], best_gt['cy'], 0.15, best_gt['yaw'], T_world2cam, intrinsics, thickness=5)
            draw_pose_axes(vis, est_cx, est_cy, 0.15, est_yaw, T_world2cam, intrinsics, thickness=2)
            cv2.imwrite(os.path.join(img_dir, f"pallet_{m_idx}.png"), vis)

    if global_results:
        print(f"Mean ADD-S: {pd.DataFrame(global_results)['add_s_cm'].mean():.2f} cm")

if __name__ == "__main__":
    main()
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
YOLO_WEIGHTS = "/nethome/rdubey36/poseEst/large_100_v1/best.pt"
RESULTS_DIR = "results_yolo_detailed"

IMAGE_IDS = ['00000', '00001', '00003', '00004', '00005', '00010', '00020', '00025']

INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
POSES_PATH = os.path.join(BASE_DIR, "rigid_body_poses.csv")

MAX_DIST = 5.0 # Ignore background pallets

PALLET_DIMS = {
    'BlockPallet': {'hx': 0.6, 'hy': 0.4, 'hz': 0.075},
    'RackablePallet': {'hx': 0.6, 'hy': 0.5, 'hz': 0.075},
    'DEFAULT': {'hx': 0.6, 'hy': 0.4, 'hz': 0.075}
}

# 2. HELPER FUNCTIONS

def get_transform_matrix(pos, quat):
    # [w, x, y, z] to [x, y, z, w] for scipy
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = pos
    return mat

def project_3d_to_pixel(pts_world, T_world2cam, fx, fy, cx, cy):
    pts_cam = (T_world2cam[:3, :3] @ pts_world.T).T + T_world2cam[:3, 3]
    x_cv, y_cv, z_cv = -pts_cam[:, 1], -pts_cam[:, 2], pts_cam[:, 0]
    u = (x_cv / z_cv) * fx + cx
    v = (y_cv / z_cv) * fy + cy
    return np.vstack((u, v)).T

def draw_point_3d(img, pt_3d, T_world2cam, intrinsics, color, radius=8):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    pixel = project_3d_to_pixel(np.array([pt_3d]), T_world2cam, fx, fy, cx, cy)[0]
    cv2.circle(img, (int(pixel[0]), int(pixel[1])), radius, color, -1)

def draw_3d_obb(img, cx, cy, yaw, hx, hy, height, T_world2cam, intrinsics, color):
    fx, fy, cx_p, cy_p = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, -s],[s, c]])
    corners_2d = (R_yaw @ np.array([[hx,hy],[hx,-hy],[-hx,-hy],[-hx,hy]]).T).T + [cx, cy]
    pts_3d = np.vstack([np.column_stack([corners_2d, np.zeros(4)]), np.column_stack([corners_2d, np.full(4, height)])])
    pix = project_3d_to_pixel(pts_3d, T_world2cam, fx, fy, cx_p, cy_p).astype(int)
    for i in range(4):
        cv2.line(img, tuple(pix[i]), tuple(pix[(i+1)%4]), color, 2)
        cv2.line(img, tuple(pix[i+4]), tuple(pix[((i+1)%4)+4]), color, 2)
        cv2.line(img, tuple(pix[i]), tuple(pix[i+4]), color, 2)

def draw_pose_axes(img, x, y, z, yaw, T_world2cam, intrinsics, length=0.4, thickness=3):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    origin = np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([origin, origin+[c*length, s*length, 0], origin+[-s*length, c*length, 0], origin+[0,0,length]])
    pix = project_3d_to_pixel(pts, T_world2cam, fx, fy, cx, cy).astype(int)
    cv2.arrowedLine(img, tuple(pix[0]), tuple(pix[1]), (0,0,255), thickness, tipLength=0.2)
    cv2.arrowedLine(img, tuple(pix[0]), tuple(pix[2]), (0,255,0), thickness, tipLength=0.2)
    cv2.arrowedLine(img, tuple(pix[0]), tuple(pix[3]), (255,0,0), thickness, tipLength=0.2)

def calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy):
    base = np.array(list(itertools.product([hx,-hx],[hy,-hy],[0.075,-0.075])))
    def tr(pts, x, y, yaw):
        m = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0,0,1]])
        return (m @ pts.T).T + [x, y, 0.075]
    gt_c = tr(base, gt_cx, gt_cy, gt_yaw)
    return min(np.linalg.norm(gt_c - tr(base, est_cx, est_cy, est_yaw), axis=1).mean(),
               np.linalg.norm(gt_c - tr(base, est_cx, est_cy, est_yaw+np.pi), axis=1).mean())

# 3. MAIN PIPELINE

def main():
    print("Initializing YOLO Detailed Evaluation...")
    model = YOLO(YOLO_WEIGHTS)
    
    with open(INTRINSICS_PATH, 'r') as f:
        cam_data = json.load(f)
        intrinsics = {'fx': cam_data['intrinsic_matrix'][0][0], 'fy': cam_data['intrinsic_matrix'][1][1],
                      'cx': cam_data['intrinsic_matrix'][0][2], 'cy': cam_data['intrinsic_matrix'][1][2]}

    views_df = pd.read_csv(VIEWS_PATH)
    poses_df = pd.read_csv(POSES_PATH)
    global_results = []

    for img_id in IMAGE_IDS:
        print(f"\n[{img_id}] Loading Data...")
        img_bgr = cv2.imread(os.path.join(BASE_DIR, f"rgb/rgb_{img_id}.png"))
        depth = np.load(os.path.join(BASE_DIR, f"depth/depth_{img_id}.npy"))
        
        # Setup Camera
        v_row = views_df[views_df['rgb_path'].str.contains(img_id)].iloc[0]
        T_cam2world = get_transform_matrix([v_row['forklift_pos_x'], v_row['forklift_pos_y'], v_row['forklift_pos_z']],
                                           [v_row['forklift_quat_w'], v_row['forklift_quat_x'], v_row['forklift_quat_y'], v_row['forklift_quat_z']]) \
                      @ get_transform_matrix([v_row['cam_pos_x'], v_row['cam_pos_y'], v_row['cam_pos_z']],
                                           [v_row['cam_quat_w'], v_row['cam_quat_x'], v_row['cam_quat_y'], v_row['cam_quat_z']])
        T_world2cam = np.linalg.inv(T_cam2world)

        # Parse ALL Ground Truths for Pixel Matching
        p_row = poses_df.iloc[int(img_id)]
        gt_frame_data = []
        for i in range(5):
            name = str(p_row[f'pallet_{i}_name'])
            if 'Pallet' not in name: continue
            dims = PALLET_DIMS['RackablePallet'] if 'Rackable' in name else PALLET_DIMS['BlockPallet']
            q = [p_row[f'pallet_{i}_quat_w'], p_row[f'pallet_{i}_quat_x'], p_row[f'pallet_{i}_quat_y'], p_row[f'pallet_{i}_quat_z']]
            yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('zyx')[0]
            
            # Calculate GT Front Center: Center - (R * [hx, 0, 0])
            gt_cx, gt_cy = p_row[f'pallet_{i}_pos_x'], p_row[f'pallet_{i}_pos_y']
            gt_f_x = gt_cx - np.cos(yaw) * dims['hx']
            gt_f_y = gt_cy - np.sin(yaw) * dims['hx']
            
            # Project geometric center to pixels for matching
            pix_c = project_3d_to_pixel(np.array([[gt_cx, gt_cy, 0.075]]), T_world2cam, **intrinsics)[0]
            
            gt_frame_data.append({'cx': gt_cx, 'cy': gt_cy, 'yaw': yaw, 'hx': dims['hx'], 'hy': dims['hy'], 
                                  'fx': gt_f_x, 'fy': gt_f_y, 'px': pix_c[0], 'py': pix_c[1]})

        # Run YOLO
        results = model(img_bgr, conf=0.4, verbose=False)[0]
        if results.masks is None: continue

        img_dir = os.path.join(RESULTS_DIR, img_id)
        os.makedirs(img_dir, exist_ok=True)
        pallet_counter = 1

        for mask_obj in results.masks.data.cpu().numpy():
            mask = cv2.resize(mask_obj, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
            v, u = np.where(mask)
            if len(u) < 20: continue

            # --- PIXEL MATCHING ---
            # Link this YOLO mask to the correct GT based on where the GT center projects
            best_gt = None
            for gt in gt_frame_data:
                if (u.min() <= gt['px'] <= u.max()) and (v.min() <= gt['py'] <= v.max()):
                    best_gt = gt
                    break
            if best_gt is None: continue
            
            z_vals = depth[v, u]
            if np.median(z_vals) > MAX_DIST: continue

            pallet_dir = os.path.join(img_dir, str(pallet_counter))
            os.makedirs(pallet_dir, exist_ok=True)

            # --- STEP 1: SEGMENTATION ---
            vis_1 = img_bgr.copy()
            vis_1[mask] = [255, 0, 0]
            cv2.addWeighted(img_bgr, 0.6, vis_1, 0.4, 0, dst=vis_1)
            cv2.imwrite(os.path.join(pallet_dir, "01_segmentation_mask.png"), vis_1)

            # --- 3D EXTRACTION & RANSAC ---
            x_cv, y_cv = (u - intrinsics['cx']) * z_vals / intrinsics['fx'], (v - intrinsics['cy']) * z_vals / intrinsics['fy']
            pts_world = (T_cam2world @ np.vstack((z_vals, -x_cv, -y_cv, np.ones_like(z_vals))).T.T).T[:, :3]

            time_start = time.perf_counter()
            front_pts = pts_world[pts_world[:, 0] < (np.min(pts_world[:, 0]) + 0.15)]
            X, Y = front_pts[:, 0], front_pts[:, 1]
            ransac = RANSACRegressor(residual_threshold=0.03).fit(Y.reshape(-1,1), X)
            
            m = ransac.estimator_.coef_[0]
            est_yaw = np.arctan2(-m, 1)
            est_f_y = np.mean(Y[ransac.inlier_mask_])
            est_f_x = ransac.predict([[est_f_y]])[0]
            
            # Geometric Center calculation
            est_cx = est_f_x + np.cos(est_yaw) * best_gt['hx']
            est_cy = est_f_y + np.sin(est_yaw) * best_gt['hx']
            time_end = time.perf_counter()

            # --- STEP 2 & 3: PLOTS ---
            plt.figure(figsize=(6,6))
            plt.scatter(pts_world[:,0], pts_world[:,1], s=5, c='blue')
            plt.axis('equal'); plt.grid(True); plt.savefig(os.path.join(pallet_dir, "02_3d_points.png")); plt.close()

            plt.figure(figsize=(6,6))
            plt.scatter(Y[~ransac.inlier_mask_], X[~ransac.inlier_mask_], c='red', s=10)
            plt.scatter(Y[ransac.inlier_mask_], X[ransac.inlier_mask_], c='green', s=10)
            plt.plot([Y.min(), Y.max()], [m*Y.min()+ransac.estimator_.intercept_, m*Y.max()+ransac.estimator_.intercept_], 'b')
            plt.axis('equal'); plt.grid(True); plt.savefig(os.path.join(pallet_dir, "03_ransac_fit.png")); plt.close()

            # --- STEP 4: FRONT CENTER BENCHMARK ---
            vis_4 = img_bgr.copy()
            draw_3d_obb(vis_4, best_gt['cx'], best_gt['cy'], best_gt['yaw'], best_gt['hx'], best_gt['hy'], 0.15, T_world2cam, intrinsics, (0,255,0))
            draw_point_3d(vis_4, [best_gt['fx'], best_gt['fy'], 0.05], T_world2cam, intrinsics, (0,255,0), 10)
            draw_point_3d(vis_4, [est_f_x, est_f_y, 0.05], T_world2cam, intrinsics, (0,0,255), 10)
            cv2.imwrite(os.path.join(pallet_dir, "04_front_center_benchmark.png"), vis_4)

            # --- STEP 5: 6D POSE AXES ---
            vis_5 = img_bgr.copy()
            draw_pose_axes(vis_5, best_gt['cx'], best_gt['cy'], 0.15, best_gt['yaw'], T_world2cam, intrinsics, thickness=6)
            draw_pose_axes(vis_5, est_cx, est_cy, 0.15, est_yaw, T_world2cam, intrinsics, thickness=2)
            cv2.imwrite(os.path.join(pallet_dir, "05_6d_pose_benchmark.png"), vis_5)

            # --- METRICS ---
            adds = calculate_adds(best_gt['cx'], best_gt['cy'], best_gt['yaw'], est_cx, est_cy, est_yaw, best_gt['hx'], best_gt['hy'])
            res = {'Image': img_id, 'ADD-S_cm': adds*100, 'Yaw_Err': abs(np.degrees(est_yaw-best_gt['yaw']))%180, 'Time': time_end-time_start}
            global_results.append(res)
            pallet_counter += 1

    if global_results:
        df = pd.DataFrame(global_results)
        df.to_csv(os.path.join(RESULTS_DIR, "overall_results.csv"), index=False)
        print(f"Done. Mean ADD-S: {df['ADD-S_cm'].mean():.2f} cm")

if __name__ == "__main__":
    main()
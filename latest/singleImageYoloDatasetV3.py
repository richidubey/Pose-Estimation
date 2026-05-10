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

# ==========================================
# 1. SETUP & CONFIGURATION (SINGLE IMAGE)
# ==========================================

BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv3"

IMAGE_ID = '00001'

INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
FORKLIFT_VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
POSES_PATH = os.path.join(BASE_DIR, "rigid_body_poses.csv")
LOCAL_SPECS_PATH = os.path.join(BASE_DIR, "rigid_body_local_specs.csv")

YOLO_WEIGHTS = "/nethome/rdubey36/poseEst/large_100_v1/best.pt"
DEBUG_DIR = "debug_results"


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
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    origin = np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    
    pt_x = origin + np.array([c, s, 0]) * length
    pt_y = origin + np.array([-s, c, 0]) * length
    pt_z = origin + np.array([0, 0, length])
    
    pts_3d = np.array([origin, pt_x, pt_y, pt_z])
    pixels = project_3d_to_pixel(pts_3d, T_world2cam, fx, fy, cx, cy)
    if len(pixels) == 4:
        p_org, p_x, p_y, p_z =[tuple(p.astype(int)) for p in pixels]
        cv2.arrowedLine(img, p_org, p_x, (0, 0, 255), thickness, tipLength=0.2)
        cv2.arrowedLine(img, p_org, p_y, (0, 255, 0), thickness, tipLength=0.2)
        cv2.arrowedLine(img, p_org, p_z, (255, 0, 0), thickness, tipLength=0.2)

def calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy, hz):
    base_corners = np.array(list(itertools.product([hx, -hx], [hy, -hy],[hz, -hz])))
    
    def transform(corners, center_x, center_y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        R_mat = np.array([[c, -s, 0],[s, c, 0], [0, 0, 1]])
        return (R_mat @ corners.T).T + np.array([center_x, center_y, hz]) 
    
    gt_corners = transform(base_corners, gt_cx, gt_cy, gt_yaw)
    est_corners_1 = transform(base_corners, est_cx, est_cy, est_yaw)
    est_corners_2 = transform(base_corners, est_cx, est_cy, est_yaw + np.pi)
    
    dist_1 = np.linalg.norm(gt_corners - est_corners_1, axis=1).mean()
    dist_2 = np.linalg.norm(gt_corners - est_corners_2, axis=1).mean()
    return min(dist_1, dist_2)


# ==========================================
# 3. DEBUG PIPELINE
# ==========================================

def main():
    print(f"\n=======================================================")
    print(f"--- STARTING DEBUG PIPELINE FOR IMAGE: {IMAGE_ID} ---")
    print(f"=======================================================\n")
    
    print("[1] Loading YOLO Model...")
    model = YOLO(YOLO_WEIGHTS)
    
    print("[2] Loading Intrinsics and Data Tables...")
    with open(INTRINSICS_PATH, 'r') as f:
        intrinsics_data = json.load(f)
        fx, fy = intrinsics_data['intrinsic_matrix'][0][0], intrinsics_data['intrinsic_matrix'][1][1]
        cx, cy = intrinsics_data['intrinsic_matrix'][0][2], intrinsics_data['intrinsic_matrix'][1][2]
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    views_df = pd.read_csv(FORKLIFT_VIEWS_PATH)
    poses_df = pd.read_csv(POSES_PATH)
    specs_df = pd.read_csv(LOCAL_SPECS_PATH)
    specs_dict = specs_df.set_index('name').to_dict('index')

    rgb_path_str = f"rgb/rgb_{IMAGE_ID}.png"
    rgb_path = os.path.join(BASE_DIR, rgb_path_str)
    depth_path = os.path.join(BASE_DIR, f"depth/depth_{IMAGE_ID}.npy")
    
    if not os.path.exists(rgb_path):
        print(f"ERROR: Image {rgb_path} not found.")
        return
        
    if not (views_df['rgb_path'] == rgb_path_str).any():
        print(f"ERROR: No entry in views_df for {rgb_path_str}.")
        return

    print("[3] Loading Image and Depth Maps...")
    img = cv2.imread(rgb_path)
    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    
    row = views_df[views_df['rgb_path'] == rgb_path_str].iloc[0]
    sample_id = int(row['sample_id'])
    gt_row = poses_df.iloc[sample_id]
    
    # Pose transforms
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
    cam_world_pos = T_cam2world[:3, 3]

    print("[4] Running YOLO Inference...")
    results = model(img_rgb, verbose=False)
    result = results[0]
    
    yolo_box_count = len(result.boxes) if result.boxes is not None else 0
    print(f"    -> YOLO found {yolo_box_count} total objects in scene.")
    
    img_dir = os.path.join(DEBUG_DIR, IMAGE_ID)
    os.makedirs(img_dir, exist_ok=True)

    pallet_cols =[col for col in gt_row.index if str(col).startswith('pallet_') and str(col).endswith('_name')]
    
    pallet_counter = 1
    image_results =[]

    print(f"\n[5] Parsing Ground Truth Pallets...")
    for name_col in pallet_cols:
        if pd.isna(gt_row[name_col]):
            continue
        
        pallet_name = gt_row[name_col]
        if not isinstance(pallet_name, str) or 'Pallet' not in pallet_name:
            continue

        print(f"\n--- Evaluating Pallet {pallet_counter} ({pallet_name}) ---")
        prefix = name_col.replace('_name', '')
        gt_cx = gt_row[f'{prefix}_pos_x']
        gt_cy = gt_row[f'{prefix}_pos_y']
        
        qw, qx, qy, qz = gt_row[f'{prefix}_quat_w'], gt_row[f'{prefix}_quat_x'], gt_row[f'{prefix}_quat_y'], gt_row[f'{prefix}_quat_z']
        
        r = R.from_quat([qx, qy, qz, qw])
        R_mat_3d = r.as_matrix()
        gt_yaw = np.arctan2(R_mat_3d[1, 0], R_mat_3d[0, 0])
        
        spec = specs_dict.get(pallet_name, None)
        if spec is None:
            print("    [!] Missing dimensions in local specs. Skipping.")
            continue
            
        hx, hy, hz = spec['span_x'] / 2.0, spec['span_y'] / 2.0, spec['span_z'] / 2.0
        c_cos, c_sin = np.cos(gt_yaw), np.sin(gt_yaw)
        R_mat = np.array([[c_cos, -c_sin],[c_sin, c_cos]])
        
        # Calculate true GT Front Edge based on distance to camera
        local_midpoints = np.array([[hx, 0.0], [-hx, 0.0], [0.0, hy], [0.0, -hy]])
        world_midpoints = (R_mat @ local_midpoints.T).T + np.array([gt_cx, gt_cy])
        
        cam_xy = cam_world_pos[:2]
        dists_to_cam = np.linalg.norm(world_midpoints - cam_xy, axis=1)
        closest_idx = np.argmin(dists_to_cam)
        
        gt_front_x, gt_front_y = world_midpoints[closest_idx]
        pallet_depth_radius = hx if closest_idx in [0, 1] else hy
        
        gt_normal = np.array([gt_cx - gt_front_x, gt_cy - gt_front_y])
        gt_normal_len = np.linalg.norm(gt_normal)
        if gt_normal_len > 0:
            gt_normal = gt_normal / gt_normal_len
        else:
            gt_normal = np.array([np.cos(gt_yaw), np.sin(gt_yaw)])

        dist_to_cam = np.linalg.norm(cam_world_pos - np.array([gt_cx, gt_cy, 0.0]))
        
        # Project GT 3D Front Point to 2D
        pt_3d = np.array([[gt_front_x, gt_front_y, hz]])
        pts_cam_body = (T_world2cam[:3, :3] @ pt_3d.T).T + T_world2cam[:3, 3]
        
        if pts_cam_body[0, 0] <= 0:
            print("    [!] GT Pallet is behind the camera. Skipping.")
            continue 
            
        pixel = project_3d_to_pixel(pt_3d, T_world2cam, fx, fy, cx, cy)[0]
        px, py = int(pixel[0]), int(pixel[1])

        print(f"    GT Center 3D:    ({gt_cx:.2f}, {gt_cy:.2f})")
        print(f"    GT Front 3D:     ({gt_front_x:.2f}, {gt_front_y:.2f})")
        print(f"    GT Projected 2D: (u: {px}, v: {py}) | Dist: {dist_to_cam:.2f}m")

        if not (0 <= px < img_w and 0 <= py < img_h):
            print("    [!] GT projection is out of bounds (off-screen). Skipping.")
            continue 

        # --- YOLO DETECTION MATCHING ---
        matched_idx = -1
        best_dist = 200 
        
        if yolo_box_count > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            for i_box, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = np.hypot(px - box_cx, py - box_cy)
                print(f"      -> Distance to YOLO Box {i_box}: {dist:.1f} pixels")
                if dist < best_dist:
                    best_dist = dist
                    matched_idx = i_box

        if matched_idx == -1:
            print(f"    [!] YOLO completely missed this pallet (Threshold = 200px). Skipping.")
            continue

        print(f"    [MATCH SUCCESS] Matched with YOLO Box {matched_idx} (Dist: {best_dist:.1f}px)")
        
        pallet_dir = os.path.join(img_dir, str(pallet_counter))
        os.makedirs(pallet_dir, exist_ok=True)

        # --- STEP 1: SEGMENTATION / EXTRACTION ---
        point_coords = np.array([[px, py]])
        mask = np.zeros((img_h, img_w), dtype=bool)
        
        if result.masks is not None:
            polygon = result.masks.xy[matched_idx]
            single_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(single_mask, [pts], 255)
            mask = single_mask == 255
        else:
            x1, y1, x2, y2 = result.boxes.xyxy[matched_idx].cpu().numpy().astype(int)
            mask[y1:y2, x1:x2] = True

        vis_step1 = img.copy()
        vis_step1[mask] =[255, 0, 0] 
        vis_step1 = cv2.addWeighted(img, 0.6, vis_step1, 0.4, 0)
        cv2.circle(vis_step1, tuple(point_coords[0]), 5, (0, 0, 255), -1) 
        cv2.imwrite(os.path.join(pallet_dir, "01_segmentation_mask.png"), vis_step1)

        # --- EXTRACT FRONT EDGE ---
        v_coords, u_coords = np.where(mask)
        if len(u_coords) == 0:
            print("    [!] Mask generation failed (0 pixels). Skipping.")
            continue
            
        df_mask = pd.DataFrame({'u': u_coords, 'v': v_coords})
        bottom_edge = df_mask.groupby('u')['v'].max().reset_index()
        u, v = bottom_edge['u'].values, bottom_edge['v'].values
        z = depth[v, u]
        
        valid = (z > 0) & (z < 15.0)
        u, v, z = u[valid], v[valid], z[valid]
        print(f"    Depth points extracted from bottom edge: {len(z)} points")

        if len(z) < 3:
            print("    [!] Not enough valid depth points (<3) after filtering. Skipping.")
            continue

        x_cv, y_cv = (u - cx) * z / fx, (v - cy) * z / fy
        pts_body = np.vstack((z, -x_cv, -y_cv, np.ones_like(z))).T
        front_pts_world = (T_cam2world @ pts_body.T).T[:, :3]

        # START TIMING
        time_start = time.perf_counter()

        front_pts_world_trunc = front_pts_world
        #Not truncating with a fixed number anymore due to rotation considerations.

        # RANSAC
        X, Y = front_pts_world_trunc[:, 0], front_pts_world_trunc[:, 1]
        try:
            ransac = RANSACRegressor(min_samples=3, residual_threshold=0.03) 
            ransac.fit(Y.reshape(-1, 1), X)
        except ValueError as e:
            print(f"    [!] RANSAC Math Error: {e}. Skipping.")
            continue
        
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_
        inlier_mask = ransac.inlier_mask_
        inlier_Y = Y[inlier_mask]
        print(f"    RANSAC Inliers found: {len(inlier_Y)} out of {len(Y)}")
        
        min_Y, max_Y = np.min(inlier_Y), np.max(inlier_Y)
        est_front_y = (min_Y + max_Y) / 2.0
        est_front_x = m * est_front_y + c 
        
        n = np.array([1.0, -m])
        n = n / np.linalg.norm(n)
        if np.dot(n, gt_normal) < 0:
            n = -n
        
        est_cx = est_front_x + n[0] * pallet_depth_radius
        est_cy = est_front_y + n[1] * pallet_depth_radius
        
        angle_diff = np.arctan2(n[1], n[0]) - np.arctan2(gt_normal[1], gt_normal[0])
        est_yaw = gt_yaw + angle_diff
        est_yaw = np.arctan2(np.sin(est_yaw), np.cos(est_yaw))

        time_end = time.perf_counter()
        est_time_sec = time_end - time_start

        # Metrics
        err_x = abs(est_front_x - gt_front_x)
        err_y = abs(est_front_y - gt_front_y)
        err_yaw_deg = abs(np.degrees(est_yaw) - np.degrees(gt_yaw))
        add_s = calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy, hz)

        print(f"    [METRICS] ADD-S: {add_s*100:.2f} cm | Yaw Err: {err_yaw_deg:.2f} deg | Time: {est_time_sec*1000:.2f} ms")

        res_dict = {
            'Pallet_ID': pallet_counter,
            'Dist_to_Cam_m': round(dist_to_cam, 2),
            'Error_X_cm': round(err_x * 100, 2),
            'Error_Y_cm': round(err_y * 100, 2),
            'Error_Yaw_deg': round(err_yaw_deg, 2),
            'ADD_S_cm': round(add_s * 100, 2)
        }
        image_results.append(res_dict)

        # --- DRAW DEBUG PLOTS ---
        plt.figure(figsize=(8,8))
        plt.scatter(front_pts_world_trunc[:, 0], front_pts_world_trunc[:, 1], s=10, c='blue', label='Front Edge Points')
        plt.xlabel('X (World Coordinates)')
        plt.ylabel('Y (World Coordinates)')
        plt.title(f"Image {IMAGE_ID} - Pallet {pallet_counter} - 3D Projected Front Edge")
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
        plt.title(f"Image {IMAGE_ID} - Pallet {pallet_counter} - RANSAC Edge Fitting")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(pallet_dir, "03_ransac_fit.png"))
        plt.close()

        vis_img_4 = img.copy()  
        draw_3d_obb(vis_img_4, gt_cx, gt_cy, gt_yaw, hx, hy, hz * 2, T_world2cam, intrinsics, color=(0, 255, 0))
        draw_point_3d(vis_img_4,[gt_front_x, gt_front_y, hz], T_world2cam, intrinsics, color=(0, 255, 0), radius=10)
        draw_point_3d(vis_img_4,[est_front_x, est_front_y, hz], T_world2cam, intrinsics, color=(0, 0, 255), radius=10)
        text_block_4 =[
            f"PALLET {pallet_counter} | Dist to Cam: {dist_to_cam:.2f} m",
            f"Front Center X Error: {err_x*100:.2f} cm",
            f"Front Center Y Error: {err_y*100:.2f} cm",
            f"Yaw Error: {err_yaw_deg:.2f} deg"
        ]
        for i, line in enumerate(text_block_4 +["", "GREEN Dot: GT | RED Dot: RANSAC"]):
            cv2.putText(vis_img_4, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(pallet_dir, "04_front_center_benchmark.png"), vis_img_4)

        vis_img_5 = img.copy()
        draw_pose_axes(vis_img_5, gt_cx, gt_cy, hz * 2, gt_yaw, T_world2cam, intrinsics, thickness=6, length=0.4)
        draw_pose_axes(vis_img_5, est_cx, est_cy, hz * 2, est_yaw, T_world2cam, intrinsics, thickness=2, length=0.4)
        text_block_5 =[
            f"PALLET {pallet_counter} | Dist to Cam: {dist_to_cam:.2f} m",
            f"ADD-S Benchmark: {add_s*100:.2f} cm"
        ]
        for i, line in enumerate(text_block_5 +["", "THICK Axes: GT | THIN Axes: RANSAC Est"]):
            cv2.putText(vis_img_5, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(pallet_dir, "05_6d_pose_benchmark.png"), vis_img_5)

        pallet_counter += 1

    print(f"\n=======================================================")
    if len(image_results) > 0:
        df_img = pd.DataFrame(image_results)
        print(f"FINISHED PROCESSING {len(image_results)} PALLETS.")
        print("SUMMARY OF ERRORS:")
        print(df_img[['Pallet_ID', 'Dist_to_Cam_m', 'Error_Yaw_deg', 'ADD_S_cm']].to_string(index=False))
        print(f"\nSaved plots and logs to: {img_dir}/")
    else:
        print("FINISHED. No pallets were successfully fully processed.")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
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
# 1. SETUP & CONFIGURATION
# ==========================================

BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv3"

INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
FORKLIFT_VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
POSES_PATH = os.path.join(BASE_DIR, "rigid_body_poses.csv")
LOCAL_SPECS_PATH = os.path.join(BASE_DIR, "rigid_body_local_specs.csv")

YOLO_WEIGHTS = "/nethome/rdubey36/poseEst/large_100_v1/best.pt"
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
# 3. MAIN PIPELINE
# ==========================================

def main():
    print("Initializing YOLO Model...")
    model = YOLO(YOLO_WEIGHTS)
    
    with open(INTRINSICS_PATH, 'r') as f:
        intrinsics_data = json.load(f)
        fx, fy = intrinsics_data['intrinsic_matrix'][0][0], intrinsics_data['intrinsic_matrix'][1][1]
        cx, cy = intrinsics_data['intrinsic_matrix'][0][2], intrinsics_data['intrinsic_matrix'][1][2]
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    views_df = pd.read_csv(FORKLIFT_VIEWS_PATH)
    poses_df = pd.read_csv(POSES_PATH)
    specs_df = pd.read_csv(LOCAL_SPECS_PATH)
    specs_dict = specs_df.set_index('name').to_dict('index')

    global_results =[]
    
    rgb_dir = os.path.join(BASE_DIR, "rgb")
    image_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(".png")])
    IMAGE_IDS =[f.replace("rgb_", "").replace(".png", "") for f in image_files]
    
    print(f"Found {len(IMAGE_IDS)} total images to process.")

    for img_id in IMAGE_IDS:
        print(f"\n=======================================================")
        print(f"[{img_id}] Loading Data...")
        
        rgb_path_str = f"rgb/rgb_{img_id}.png"
        rgb_path = os.path.join(BASE_DIR, rgb_path_str)
        depth_path = os.path.join(BASE_DIR, f"depth/depth_{img_id}.npy")
        
        if not os.path.exists(rgb_path) or not (views_df['rgb_path'] == rgb_path_str).any():
            continue
            
        img = cv2.imread(rgb_path)
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        
        row = views_df[views_df['rgb_path'] == rgb_path_str].iloc[0]
        sample_id = int(row['sample_id'])
        gt_row = poses_df.iloc[sample_id]
        
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

        results = model(img_rgb, verbose=False)
        result = results[0]
        
        img_dir = os.path.join(RESULTS_DIR, img_id)
        os.makedirs(img_dir, exist_ok=True)

        # ---------------------------------------------------------
        # PRE-PROCESS YOLO BOXES INTO 3D WORLD COORDINATES
        # ---------------------------------------------------------
        yolo_objects =[]
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            for i_box, box in enumerate(boxes):
                mask = np.zeros((img_h, img_w), dtype=bool)
                if result.masks is not None:
                    polygon = result.masks.xy[i_box]
                    single_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(single_mask, [pts], 255)
                    mask = single_mask == 255
                else:
                    x1, y1, x2, y2 = box.astype(int)
                    mask[y1:y2, x1:x2] = True
                
                z_vals = depth[mask]
                valid_z = z_vals[(z_vals > 0.1) & (z_vals < 15.0)]
                if len(valid_z) == 0: continue
                median_z = np.median(valid_z)
                
                x1, y1, x2, y2 = box
                box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                x_cv = (box_cx - cx) * median_z / fx
                y_cv = (box_cy - cy) * median_z / fy
                
                pts_body = np.array([[median_z, -x_cv, -y_cv, 1.0]])
                yolo_3d_world = (T_cam2world @ pts_body.T).T[0, :3]
                
                yolo_objects.append({
                    'idx': i_box,
                    'mask': mask,
                    'world_3d': yolo_3d_world
                })

        pallet_counter = 1
        image_results =[]
        used_yolo_indices = set()

        pallet_cols =[col for col in gt_row.index if str(col).startswith('pallet_') and str(col).endswith('_name')]

        for name_col in pallet_cols:
            if pd.isna(gt_row[name_col]): continue
            
            pallet_name = gt_row[name_col]
            if not isinstance(pallet_name, str) or 'Pallet' not in pallet_name: continue

            prefix = name_col.replace('_name', '')
            gt_cx, gt_cy, gt_cz = gt_row[f'{prefix}_pos_x'], gt_row[f'{prefix}_pos_y'], gt_row[f'{prefix}_pos_z']
            
            if abs(gt_cz) > 0.15: continue
            
            qw, qx, qy, qz = gt_row[f'{prefix}_quat_w'], gt_row[f'{prefix}_quat_x'], gt_row[f'{prefix}_quat_y'], gt_row[f'{prefix}_quat_z']
            r = R.from_quat([qx, qy, qz, qw])
            R_mat_3d = r.as_matrix()
            gt_yaw = np.arctan2(R_mat_3d[1, 0], R_mat_3d[0, 0])
            
            spec = specs_dict.get(pallet_name, None)
            if spec is None: continue
                
            hx, hy, hz = spec['span_x'] / 2.0, spec['span_y'] / 2.0, spec['span_z'] / 2.0
            diameter = np.sqrt((hx*2)**2 + (hy*2)**2 + (hz*2)**2)
            c_cos, c_sin = np.cos(gt_yaw), np.sin(gt_yaw)
            R_mat = np.array([[c_cos, -c_sin],[c_sin, c_cos]])
            
            # ---> BUG 1 FIX: GT FRONT FACE VIA DOT PRODUCT ALIGNMENT <---
            local_midpoints = np.array([[hx, 0.0],[-hx, 0.0], [0.0, hy], [0.0, -hy]])
            local_normals = np.array([[1.0, 0.0],[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
            
            world_midpoints = (R_mat @ local_midpoints.T).T + np.array([gt_cx, gt_cy])
            world_normals = (R_mat @ local_normals.T).T
            
            vec_to_cam = cam_world_pos[:2] - np.array([gt_cx, gt_cy])
            vec_to_cam = vec_to_cam / (np.linalg.norm(vec_to_cam) + 1e-6)
            
            # Select face whose outward normal points most directly at the camera
            dots = np.dot(world_normals, vec_to_cam)
            closest_idx = np.argmax(dots)
            
            gt_front_x, gt_front_y = world_midpoints[closest_idx]
            pallet_depth_radius = hx if closest_idx in [0, 1] else hy
            
            # The inward normal points INTO the pallet body for RANSAC offset
            gt_normal_inward = -world_normals[closest_idx]

            dist_to_cam = np.linalg.norm(cam_world_pos - np.array([gt_cx, gt_cy, 0.0]))
            pt_3d = np.array([[gt_front_x, gt_front_y, hz]])

            matched_idx = -1
            best_dist_3d = 1.5  
            best_mask = None
            
            for yolo_obj in yolo_objects:
                if yolo_obj['idx'] in used_yolo_indices: continue
                dist_3d = np.linalg.norm(pt_3d[0] - yolo_obj['world_3d'])
                if dist_3d < best_dist_3d:
                    best_dist_3d = dist_3d
                    matched_idx = yolo_obj['idx']
                    best_mask = yolo_obj['mask'].copy()

            if matched_idx == -1: continue

            used_yolo_indices.add(matched_idx)

            print(f"  -> Pallet {pallet_counter} in view (Dist: {dist_to_cam:.1f}m). Matched YOLO {matched_idx}")
            pallet_dir = os.path.join(img_dir, str(pallet_counter))
            os.makedirs(pallet_dir, exist_ok=True)

            pixel = project_3d_to_pixel(pt_3d, T_world2cam, fx, fy, cx, cy)[0]
            px, py = int(pixel[0]), int(pixel[1])
            point_coords = np.array([[px, py]])
            
            # ---------------------------------------------------------
            # ---> BUG 2 FIX: HORIZONTAL BAND FILTER (TRIMMING) <---
            # ---------------------------------------------------------
            row_counts = best_mask.sum(axis=1)
            if row_counts.max() > 0:
                max_width = row_counts.max()
                # Find the primary horizontal body (rows with at least 50% max width)
                valid_rows = np.where(row_counts >= 0.5 * max_width)[0]
                
                if len(valid_rows) > 0:
                    v_min, v_max = valid_rows.min(), valid_rows.max()
                    # Add 20% margin above/below to not chop corners of rotated pallets
                    margin = max(2, int((v_max - v_min) * 0.2))
                    v_min = max(0, v_min - margin)
                    v_max = min(img_h - 1, v_max + margin)
                    
                    # Delete vertical tails (reflections) and heads (stacked boxes)
                    best_mask[:v_min, :] = False
                    best_mask[v_max+1:, :] = False

            vis_step1 = img.copy()
            vis_step1[best_mask] = [255, 0, 0] 
            vis_step1 = cv2.addWeighted(img, 0.6, vis_step1, 0.4, 0)
            if 0 <= px < img_w and 0 <= py < img_h:
                cv2.circle(vis_step1, tuple(point_coords[0]), 5, (0, 0, 255), -1) 
            cv2.imwrite(os.path.join(pallet_dir, "01_segmentation_mask.png"), vis_step1)

            # --- EXTRACT FRONT EDGE ---
            v_coords, u_coords = np.where(best_mask)
            if len(u_coords) == 0: continue
                
            df_mask = pd.DataFrame({'u': u_coords, 'v': v_coords})
            bottom_edge = df_mask.groupby('u')['v'].max().reset_index()
            u, v = bottom_edge['u'].values, bottom_edge['v'].values
            z = depth[v, u]
            
            valid = (z > 0) & (z < 15.0)
            u, v, z = u[valid], v[valid], z[valid]

            if len(z) < 3: continue

            x_cv, y_cv = (u - cx) * z / fx, (v - cy) * z / fy
            pts_body = np.vstack((z, -x_cv, -y_cv, np.ones_like(z))).T
            front_pts_world = (T_cam2world @ pts_body.T).T[:, :3]

            time_start = time.perf_counter()

            # RANSAC
            X, Y = front_pts_world[:, 0], front_pts_world[:, 1]
            var_x, var_y = np.var(X), np.var(Y)
            
            try:
                ransac = RANSACRegressor(min_samples=3, residual_threshold=0.03) 
                
                # ---------------------------------------------------------
                # ---> BUG 3 FIX: DYNAMIC RANSAC AXIS REGRESSION <---
                # ---------------------------------------------------------
                if var_x > var_y:
                    # Line is spread along X axis, fit Y as function of X
                    ransac.fit(X.reshape(-1, 1), Y)
                    m, c = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
                    inlier_mask = ransac.inlier_mask_
                    
                    min_X, max_X = np.min(X[inlier_mask]), np.max(X[inlier_mask])
                    est_front_x = (min_X + max_X) / 2.0
                    est_front_y = m * est_front_x + c
                    
                    n = np.array([-m, 1.0])
                else:
                    # Line is spread along Y axis, fit X as function of Y
                    ransac.fit(Y.reshape(-1, 1), X)
                    m, c = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
                    inlier_mask = ransac.inlier_mask_
                    
                    min_Y, max_Y = np.min(Y[inlier_mask]), np.max(Y[inlier_mask])
                    est_front_y = (min_Y + max_Y) / 2.0
                    est_front_x = m * est_front_y + c
                    
                    n = np.array([1.0, -m])
                    
            except ValueError:
                continue
            
            # Normalize RANSAC vector
            n = n / np.linalg.norm(n)
            
            # Force vector to point INWARD towards the pallet center
            if np.dot(n, gt_normal_inward) < 0:
                n = -n
            
            est_cx = est_front_x + n[0] * pallet_depth_radius
            est_cy = est_front_y + n[1] * pallet_depth_radius
            
            angle_diff = np.arctan2(n[1], n[0]) - np.arctan2(gt_normal_inward[1], gt_normal_inward[0])
            est_yaw = gt_yaw + angle_diff
            est_yaw = np.arctan2(np.sin(est_yaw), np.cos(est_yaw))

            time_end = time.perf_counter()
            est_time_sec = time_end - time_start

            # Metrics
            err_x = abs(est_front_x - gt_front_x)
            err_y = abs(est_front_y - gt_front_y)
            err_yaw_deg = abs(np.degrees(est_yaw) - np.degrees(gt_yaw))
            add_s = calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy, hz)
            
            threshold_01d = 0.1 * diameter
            adds_pass_01d = bool(add_s < threshold_01d)

            res_dict = {
                'Image_ID': img_id,
                'Pallet_ID': pallet_counter,
                'Dist_to_Cam_m': round(dist_to_cam, 2),
                'Error_X_cm': round(err_x * 100, 2),
                'Error_Y_cm': round(err_y * 100, 2),
                'Error_Yaw_deg': round(err_yaw_deg, 2),
                'ADD_S_cm': round(add_s * 100, 2),
                'Diameter_m': round(diameter, 3),
                'ADD_S_Pass_01d': adds_pass_01d,
                'Time_sec': est_time_sec
            }
            image_results.append(res_dict)
            global_results.append(res_dict)

            # --- STEP 2: 3D FRONT EDGE PLOT ---
            plt.figure(figsize=(8,8))
            plt.scatter(front_pts_world[:, 0], front_pts_world[:, 1], s=10, c='blue', label='Front Edge Points')
            plt.xlabel('X (World Coordinates)')
            plt.ylabel('Y (World Coordinates)')
            plt.title(f"Image {img_id} - Pallet {pallet_counter} - 3D Projected Front Edge")
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(pallet_dir, "02_3d_front_edge.png"))
            plt.close()

            # --- STEP 3: RANSAC FIT PLOT ---
            plt.figure(figsize=(8,8))
            outliers = np.logical_not(inlier_mask)
            plt.scatter(X[inlier_mask], Y[inlier_mask], s=15, c='green', label='RANSAC Inliers')
            plt.scatter(X[outliers], Y[outliers], s=15, c='red', label='Outliers')
            
            if var_x > var_y:
                line_X = np.array([X.min() - 0.1, X.max() + 0.1])
                line_Y = m * line_X + c
            else:
                line_Y = np.array([Y.min() - 0.1, Y.max() + 0.1])
                line_X = m * line_Y + c
                
            plt.plot(line_X, line_Y, color='blue', linewidth=2, label='Fitted Line')
            plt.scatter([est_front_x], [est_front_y], c='purple', marker='X', s=150, label='Estimated Front Center')
            plt.xlabel('X (World Coordinates)')
            plt.ylabel('Y (World Coordinates)')
            plt.title(f"Image {img_id} - Pallet {pallet_counter} - RANSAC Edge Fitting")
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(pallet_dir, "03_ransac_fit.png"))
            plt.close()

            # --- STEP 4: FRONT CENTER BENCHMARK ---
            vis_img_4 = img.copy()  
            draw_3d_obb(vis_img_4, gt_cx, gt_cy, gt_yaw, hx, hy, hz * 2, T_world2cam, intrinsics, color=(0, 255, 0))
            if 0 <= px < img_w and 0 <= py < img_h:
                draw_point_3d(vis_img_4,[gt_front_x, gt_front_y, hz], T_world2cam, intrinsics, color=(0, 255, 0), radius=10)
            draw_point_3d(vis_img_4, [est_front_x, est_front_y, hz], T_world2cam, intrinsics, color=(0, 0, 255), radius=10)
            text_block_4 =[
                f"PALLET {pallet_counter} | Dist to Cam: {dist_to_cam:.2f} m",
                f"Front Center X Error: {err_x*100:.2f} cm",
                f"Front Center Y Error: {err_y*100:.2f} cm",
                f"Yaw Error: {err_yaw_deg:.2f} deg"
            ]
            for i, line in enumerate(text_block_4 +["", "GREEN Dot: GT | RED Dot: RANSAC"]):
                cv2.putText(vis_img_4, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(pallet_dir, "04_front_center_benchmark.png"), vis_img_4)

            # --- STEP 5: 6D POSE BENCHMARK (AXES) ---
            vis_img_5 = img.copy()
            draw_pose_axes(vis_img_5, gt_cx, gt_cy, hz * 2, gt_yaw, T_world2cam, intrinsics, thickness=6, length=0.4)
            draw_pose_axes(vis_img_5, est_cx, est_cy, hz * 2, est_yaw, T_world2cam, intrinsics, thickness=2, length=0.4)
            pass_status = "PASS" if adds_pass_01d else "FAIL"
            text_block_5 =[
                f"PALLET {pallet_counter} | Dist to Cam: {dist_to_cam:.2f} m",
                f"ADD-S Error: {add_s*100:.2f} cm",
                f"ADD-S (< 0.1d): {pass_status}",
                f"Inference Time: {est_time_sec*1000:.2f} ms"
            ]
            for i, line in enumerate(text_block_5 +["", "THICK Axes: GT | THIN Axes: RANSAC Est"]):
                cv2.putText(vis_img_5, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(pallet_dir, "05_6d_pose_benchmark.png"), vis_img_5)

            pallet_counter += 1

        # Generate Per-Image Report
        if len(image_results) > 0:
            df_img = pd.DataFrame(image_results)
            df_img.to_csv(os.path.join(img_dir, "benchmark_data.csv"), index=False)
            
            best_p = df_img.loc[df_img['ADD_S_cm'].idxmin()]
            worst_p = df_img.loc[df_img['ADD_S_cm'].idxmax()]
            
            with open(os.path.join(img_dir, "benchmark_summary.txt"), 'w') as f:
                f.write(f"--- BENCHMARK REPORT FOR IMAGE {img_id} ---\n")
                f.write(f"Total Floor Pallets Processed: {len(df_img)}\n")
                f.write(f"Average ADD-S Error: {df_img['ADD_S_cm'].mean():.2f} cm\n")
                f.write(f"Average RANSAC Time: {df_img['Time_sec'].mean()*1000:.2f} ms\n\n")
                f.write(f"🏆 BEST PERFORMING PALLET: ID {best_p['Pallet_ID']} (Dist: {best_p['Dist_to_Cam_m']}m) | ADD-S: {best_p['ADD_S_cm']:.2f} cm\n")
                f.write(f"⚠️ WORST PERFORMING PALLET: ID {worst_p['Pallet_ID']} (Dist: {worst_p['Dist_to_Cam_m']}m) | ADD-S: {worst_p['ADD_S_cm']:.2f} cm\n")

    # 4. GLOBAL AGGREGATE REPORT
    if len(global_results) > 0:
        df_global = pd.DataFrame(global_results)
        df_global.to_csv(os.path.join(RESULTS_DIR, "global_benchmark_data.csv"), index=False)
        
        best_g = df_global.loc[df_global['ADD_S_cm'].idxmin()]
        worst_g = df_global.loc[df_global['ADD_S_cm'].idxmax()]
        
        total_pallets = len(df_global)
        pallets_passed = df_global['ADD_S_Pass_01d'].sum()
        adds_accuracy_pct = (pallets_passed / total_pallets) * 100.0 if total_pallets > 0 else 0
        
        with open(os.path.join(RESULTS_DIR, "GLOBAL_BENCHMARK_SUMMARY.txt"), 'w') as f:
            f.write("===================================================\n")
            f.write("          OVERALL 6D POSE BENCHMARK REPORT         \n")
            f.write("===================================================\n\n")
            f.write(f"Total Images Processed:    {len(IMAGE_IDS)}\n")
            f.write(f"Total Pallets Evaluated:   {total_pallets}\n\n")
            
            f.write("--- ACADEMIC SUCCESS METRICS ---\n")
            f.write(f"ADD-S (< 0.1d) Accuracy:   {adds_accuracy_pct:.2f} %\n\n")
            
            f.write("--- CONTINUOUS ERROR METRICS ---\n")
            f.write(f"Average ADD-S Error:       {df_global['ADD_S_cm'].mean():.2f} cm\n")
            f.write(f"Average X Translation:     {df_global['Error_X_cm'].mean():.2f} cm\n")
            f.write(f"Average Y Translation:     {df_global['Error_Y_cm'].mean():.2f} cm\n")
            f.write(f"Average Yaw Rotation:      {df_global['Error_Yaw_deg'].mean():.2f} deg\n\n")
            
            f.write("--- LATENCY (RANSAC + Pose Math Only) ---\n")
            f.write(f"Total Processing Time:     {df_global['Time_sec'].sum():.4f} seconds\n")
            f.write(f"Average Time Per Pallet:   {df_global['Time_sec'].mean()*1000:.2f} milliseconds\n\n")

            f.write("--- EXTREMES ---\n")
            f.write(f"🏆 BEST OVERALL: Image {best_g['Image_ID']} | Pallet {best_g['Pallet_ID']}\n")
            f.write(f"    -> ADD-S: {best_g['ADD_S_cm']:.2f} cm (Distance: {best_g['Dist_to_Cam_m']} m)\n\n")
            
            f.write(f"⚠️ WORST OVERALL: Image {worst_g['Image_ID']} | Pallet {worst_g['Pallet_ID']}\n")
            f.write(f"    -> ADD-S: {worst_g['ADD_S_cm']:.2f} cm (Distance: {worst_g['Dist_to_Cam_m']} m)\n")

        print("\n=======================================================")
        print(f"BATCH COMPLETE! Processed {total_pallets} total pallets.")
        print(f"OVERALL ADD-S ACCURACY (< 0.1d): {adds_accuracy_pct:.2f} %")
        print(f"Check 'results/GLOBAL_BENCHMARK_SUMMARY.txt' for the final report.")
        print("=======================================================")

if __name__ == "__main__":
    main()
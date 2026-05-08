import os
import cv2
import json
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from PIL import Image

import torch
from ultralytics import YOLO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. SETUP & CONFIGURATION

BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv2/ForkliftScene3Dv2/"

# Run only on a single image and text prompt as requested
IMAGE_ID = '00000'
# TEXT_PROMPT = "the pallet closest to the camera" # e.g. "the second pallet from the right"
# TEXT_PROMPT = "the second pallet from the left"
TEXT_PROMPT = "the pallet in the center"

INTRINSICS_PATH = os.path.join(BASE_DIR, "camera_intrinsics.json")
FORKLIFT_VIEWS_PATH = os.path.join(BASE_DIR, "forklift_views.csv")
OBB_PATH = os.path.join(BASE_DIR, "rigid_body_obbs.csv")

YOLO_WEIGHTS = "/nethome/rdubey36/poseEst/large_100_v1/best.pt"
RESULTS_DIR = "results_vlm"


# 2. HELPER FUNCTIONS (Kept from original)

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

def calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy, hz=0.075):
    base_corners = np.array(list(itertools.product([hx, -hx], [hy, -hy],[hz, -hz])))
    
    def transform(corners, center_x, center_y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        R_mat = np.array([[c, -s, 0],[s, c, 0], [0, 0, 1]])
        return (R_mat @ corners.T).T + np.array([center_x, center_y, 0.075]) 
    
    gt_corners = transform(base_corners, gt_cx, gt_cy, gt_yaw)
    
    est_corners_1 = transform(base_corners, est_cx, est_cy, est_yaw)
    est_corners_2 = transform(base_corners, est_cx, est_cy, est_yaw + np.pi)
    
    dist_1 = np.linalg.norm(gt_corners - est_corners_1, axis=1).mean()
    dist_2 = np.linalg.norm(gt_corners - est_corners_2, axis=1).mean()
    return min(dist_1, dist_2)


# 3. MAIN PIPELINE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Initializing Qwen2.5-VL-3B Grounding Model...")
    qwen_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_id, torch_dtype="auto", device_map=device
    )
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_id)
    
    print("[INFO] Initializing YOLO Semantic/Box Model...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    
    # Load GT metadata
    with open(INTRINSICS_PATH, 'r') as f:
        intrinsics_data = json.load(f)
        fx, fy = intrinsics_data['intrinsic_matrix'][0][0], intrinsics_data['intrinsic_matrix'][1][1]
        cx, cy = intrinsics_data['intrinsic_matrix'][0][2], intrinsics_data['intrinsic_matrix'][1][2]
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    views_df = pd.read_csv(FORKLIFT_VIEWS_PATH)
    obbs_df = pd.read_csv(OBB_PATH)
    pallets_only_df = obbs_df[obbs_df['prim_path'].str.contains('Pallet', case=False, na=False)].copy()

    print(f"\n=======================================================")
    print(f"Loading Image {IMAGE_ID}...")
    
    rgb_path = os.path.join(BASE_DIR, f"rgb/rgb_{IMAGE_ID}.png")
    depth_path = os.path.join(BASE_DIR, f"depth/depth_{IMAGE_ID}.npy")
    
    if not os.path.exists(rgb_path):
        print(f"Error: Image {rgb_path} not found.")
        return
        
    img = cv2.imread(rgb_path)
    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    
    # Construct transforms
    row = views_df[views_df['rgb_path'] == f"rgb/rgb_{IMAGE_ID}.png"].iloc[0]
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

    img_dir = os.path.join(RESULTS_DIR, IMAGE_ID)
    os.makedirs(img_dir, exist_ok=True)

    # --- STEP 1: QWEN VISUAL GROUNDING ---
    print(f"\n[1] Running VLM Grounding for Prompt: '{TEXT_PROMPT}'")
    pil_img = Image.fromarray(img_rgb)
    messages =[
        {
            "role": "user",
            "content":[
                {"type": "image", "image": pil_img},
                {"type": "text", "text": f"Detect the {TEXT_PROMPT} in the image and return its location in the form of coordinates. The format of output should be strictly like {{\"bbox_2d\":[x1, y1, x2, y2]}} without any extra text."}
            ],
        }
    ]
    
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    print(f"[VLM OUTPUT] {output_text}")
    
    # Parse the bounding box coordinates
    bbox = None
    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            bbox = data.get("bbox_2d", data.get("box_2d", None))
        except:
            pass
            
    if not bbox: # Fallback parsing
        numbers = re.findall(r'\d+', output_text)
        if len(numbers) >= 4: bbox =[int(n) for n in numbers[:4]]
            
    if not bbox:
        print("[ERROR] Could not extract bounding box from VLM. Exiting.")
        return
        
    x1, y1, x2, y2 = bbox
    print(f"    -> Extracted absolute bounding box: {bbox}")
    
    # Calculate padded crop
    pad_x = int((x2 - x1) * 0.20)
    pad_y = int((y2 - y1) * 0.20)
    crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    crop_x2, crop_y2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
    
    # Visualization 1: VLM Crop
    vis_vlm = img.copy()
    cv2.rectangle(vis_vlm, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis_vlm, "Qwen Raw BBox", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.rectangle(vis_vlm, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
    cv2.putText(vis_vlm, "Padded YOLO Crop", (crop_x1, max(0, crop_y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(os.path.join(img_dir, "01_qwen_bbox_crop.png"), vis_vlm)


    # --- STEP 2: YOLO INFERENCE ON CROP ---
    print(f"\n[2] Passing cropped region to YOLO...")
    cropped_img_bgr = img[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_img_rgb = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    cv2.imwrite(os.path.join(img_dir, "02_yolo_input_crop.png"), cropped_img_bgr)
    
    results = yolo_model(cropped_img_rgb, verbose=False)
    result = results[0]
    
    if len(result.boxes) == 0:
        print("[ERROR] YOLO found no pallets in the cropped region. Exiting.")
        return
        
    # Get highest confidence detection in the crop
    best_idx = 0 
    
    mask_crop = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype=bool)
    if result.masks is not None:
        yolo_mask = result.masks.data[best_idx].cpu().numpy()
        yolo_mask = cv2.resize(yolo_mask, (crop_x2 - crop_x1, crop_y2 - crop_y1), interpolation=cv2.INTER_NEAREST)
        mask_crop = yolo_mask > 0.5
    else:
        bx1, by1, bx2, by2 = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        mask_crop[by1:by2, bx1:bx2] = True
        
    vis_yolo_crop = cropped_img_bgr.copy()
    vis_yolo_crop[mask_crop] =[255, 0, 0]
    vis_yolo_crop = cv2.addWeighted(cropped_img_bgr, 0.6, vis_yolo_crop, 0.4, 0)
    cv2.imwrite(os.path.join(img_dir, "03_yolo_segmentation_crop.png"), vis_yolo_crop)
    
    # Map cropped mask back to original image resolution
    mask_full = np.zeros((img_h, img_w), dtype=bool)
    mask_full[crop_y1:crop_y2, crop_x1:crop_x2] = mask_crop
    
    vis_mask_full = img.copy()
    vis_mask_full[mask_full] = [255, 0, 0]
    vis_mask_full = cv2.addWeighted(img, 0.6, vis_mask_full, 0.4, 0)
    cv2.imwrite(os.path.join(img_dir, "04_segmentation_mask_full.png"), vis_mask_full)


    # --- STEP 3: GROUND TRUTH MATCHING ---
    print(f"\n[3] Finding associated Ground Truth target for evaluation...")
    vlm_cx, vlm_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    best_gt_dist = float('inf')
    best_gt_row = None
    best_gt_px, best_gt_py, best_gt_front_x, best_gt_front_y = 0, 0, 0, 0
    
    for index, gt_row in pallets_only_df.iterrows():
        gt_cx, gt_cy, gt_yaw = gt_row['cx'], gt_row['cy'], gt_row['yaw']
        hx, hy = gt_row['hx'], gt_row['hy']
        
        c_cos, c_sin = np.cos(gt_yaw), np.sin(gt_yaw)
        R_mat = np.array([[c_cos, -c_sin],[c_sin, c_cos]])
        corners_world = (R_mat @ np.array([[hx, hy],[hx, -hy], [-hx, -hy], [-hx, hy]]).T).T + np.array([gt_cx, gt_cy])
        sorted_corners = corners_world[np.argsort(corners_world[:, 0])]
        gt_front_x = (sorted_corners[0][0] + sorted_corners[1][0]) / 2.0
        gt_front_y = (sorted_corners[0][1] + sorted_corners[1][1]) / 2.0
        
        pt_3d = np.array([[gt_front_x, gt_front_y, 0.075]])
        pts_cam_body = (T_world2cam[:3, :3] @ pt_3d.T).T + T_world2cam[:3, 3]
        if pts_cam_body[0, 0] <= 0: continue
            
        pixel = project_3d_to_pixel(pt_3d, T_world2cam, fx, fy, cx, cy)[0]
        dist = np.hypot(pixel[0] - vlm_cx, pixel[1] - vlm_cy)
        
        if dist < best_gt_dist:
            best_gt_dist = dist
            best_gt_row = gt_row
            best_gt_front_x, best_gt_front_y = gt_front_x, gt_front_y
            
    if best_gt_row is None or best_gt_dist > 400:
        print("[ERROR] Could not reliably match Qwen's selection to any GT pallet!")
        return
        
    gt_cx, gt_cy, gt_yaw = best_gt_row['cx'], best_gt_row['cy'], best_gt_row['yaw']
    hx, hy = best_gt_row['hx'], best_gt_row['hy']
    dist_to_cam = np.linalg.norm(cam_world_pos - np.array([gt_cx, gt_cy, 0.0]))
    print(f"    -> Target Matched. Distance to Cam: {dist_to_cam:.2f} m")


    # --- STEP 4: RANSAC POSE ESTIMATION ---
    print(f"\n[4] Running Depth Extraction and RANSAC Pose Math...")
    v_coords, u_coords = np.where(mask_full)
    df_mask = pd.DataFrame({'u': u_coords, 'v': v_coords})
    bottom_edge = df_mask.groupby('u')['v'].max().reset_index()
    u, v = bottom_edge['u'].values, bottom_edge['v'].values
    z = depth[v, u]
    
    valid = (z > 0) & (z < 15.0)
    u, v, z = u[valid], v[valid], z[valid]
    
    x_cv, y_cv = (u - cx) * z / fx, (v - cy) * z / fy
    pts_body = np.vstack((z, -x_cv, -y_cv, np.ones_like(z))).T
    front_pts_world = (T_cam2world @ pts_body.T).T[:, :3]
    
    time_start = time.perf_counter()
    
    # Depth Truncation
    min_x = np.min(front_pts_world[:, 0])
    depth_mask = front_pts_world[:, 0] < (min_x + 0.20)
    front_pts_world_trunc = front_pts_world[depth_mask]
    
    # RANSAC
    X, Y = front_pts_world_trunc[:, 0], front_pts_world_trunc[:, 1]
    ransac = RANSACRegressor(min_samples=3, residual_threshold=0.03) 
    ransac.fit(Y.reshape(-1, 1), X)
        
    m = ransac.estimator_.coef_[0]
    c_int = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_
    inlier_Y = Y[inlier_mask]
    
    min_Y, max_Y = np.min(inlier_Y), np.max(inlier_Y)
    est_front_y = (min_Y + max_Y) / 2.0
    est_front_x = m * est_front_y + c_int 
    
    nx, ny = -1, m
    norm = np.sqrt(nx**2 + ny**2)
    nx, ny = nx/norm, ny/norm
    est_yaw = np.arctan2(-ny, -nx)

    est_cx = est_front_x + np.cos(est_yaw) * hx
    est_cy = est_front_y + np.sin(est_yaw) * hx
    
    time_end = time.perf_counter()
    est_time_sec = time_end - time_start
    
    # Benchmarking
    err_x = abs(est_front_x - best_gt_front_x)
    err_y = abs(est_front_y - best_gt_front_y)
    err_yaw_deg = abs(np.degrees(est_yaw) - np.degrees(gt_yaw))
    add_s = calculate_adds(gt_cx, gt_cy, gt_yaw, est_cx, est_cy, est_yaw, hx, hy)
    print(f"    -> Pose Computed. ADD-S Error Benchmark: {add_s*100:.2f} cm")


    # --- STEP 5: VISUALIZE AND EXPORT FINAL RESULTS ---
    print(f"\n[5] Generating Performance Benchmark Overlays...")
    
    plt.figure(figsize=(8,8))
    plt.scatter(front_pts_world_trunc[:, 0], front_pts_world_trunc[:, 1], s=10, c='blue', label='Front Edge Points')
    plt.xlabel('X (World)')
    plt.ylabel('Y (World)')
    plt.title(f"Targeted Pallet - 3D Front Edge")
    plt.axis('equal')
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(img_dir, "05_3d_front_edge.png"))
    plt.close()

    plt.figure(figsize=(8,8))
    outliers = np.logical_not(inlier_mask)
    plt.scatter(X[inlier_mask], Y[inlier_mask], s=15, c='green', label='RANSAC Inliers')
    plt.scatter(X[outliers], Y[outliers], s=15, c='red', label='Outliers')
    line_Y = np.array([Y.min() - 0.1, Y.max() + 0.1])
    line_X = m * line_Y + c_int
    plt.plot(line_X, line_Y, color='blue', linewidth=2, label='Fitted Line')
    plt.scatter([est_front_x],[est_front_y], c='purple', marker='X', s=150, label='Estimated Front Center')
    plt.xlabel('X (World)')
    plt.ylabel('Y (World)')
    plt.title(f"Targeted Pallet - RANSAC Edge Fitting")
    plt.axis('equal')
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(img_dir, "06_ransac_fit.png"))
    plt.close()

    vis_img_7 = img.copy()  
    draw_3d_obb(vis_img_7, gt_cx, gt_cy, gt_yaw, hx, hy, 0.15, T_world2cam, intrinsics, color=(0, 255, 0))
    draw_point_3d(vis_img_7,[best_gt_front_x, best_gt_front_y, 0.05], T_world2cam, intrinsics, color=(0, 255, 0), radius=10)
    draw_point_3d(vis_img_7,[est_front_x, est_front_y, 0.05], T_world2cam, intrinsics, color=(0, 0, 255), radius=10)
    text_block_7 =[
        f"Targeted Pallet | Dist to Cam: {dist_to_cam:.2f} m",
        f"Front Center X Error: {err_x*100:.2f} cm",
        f"Front Center Y Error: {err_y*100:.2f} cm",
        f"Yaw Error: {err_yaw_deg:.2f} deg"
    ]
    for i, line in enumerate(text_block_7 + ["", "GREEN Dot: GT | RED Dot: RANSAC"]):
        cv2.putText(vis_img_7, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(img_dir, "07_front_center_benchmark.png"), vis_img_7)

    vis_img_8 = img.copy()
    draw_pose_axes(vis_img_8, gt_cx, gt_cy, 0.15, gt_yaw, T_world2cam, intrinsics, thickness=6, length=0.4)
    draw_pose_axes(vis_img_8, est_cx, est_cy, 0.15, est_yaw, T_world2cam, intrinsics, thickness=2, length=0.4)
    text_block_8 =[
        f"Targeted Pallet | Dist to Cam: {dist_to_cam:.2f} m",
        f"ADD-S Benchmark: {add_s*100:.2f} cm",
        f"Inference Time: {est_time_sec*1000:.2f} ms"
    ]
    for i, line in enumerate(text_block_8 +["", "THICK Axes: GT | THIN Axes: RANSAC Est"]):
        cv2.putText(vis_img_8, line, (20, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(img_dir, "08_6d_pose_benchmark.png"), vis_img_8)

    print(f"\n=======================================================")
    print(f"PIPELINE COMPLETE! All 8 sequence images saved to: {img_dir}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
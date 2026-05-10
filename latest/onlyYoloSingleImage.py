import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_DIR = "/coc/flash5/rdubey36/datasets/ForkliftScene3Dv3"
IMAGE_ID = "00094"  # Change this to test a different image
YOLO_WEIGHTS = "/nethome/rdubey36/poseEst/large_100_v1/best.pt"
OUTPUT_PATH = f"yolo_segmentation_test_{IMAGE_ID}.png"

def main():
    print("Loading YOLO Model...")
    model = YOLO(YOLO_WEIGHTS)
    
    # 1. Load Data
    rgb_path = os.path.join(BASE_DIR, f"rgb/rgb_{IMAGE_ID}.png")
    depth_path = os.path.join(BASE_DIR, f"depth/depth_{IMAGE_ID}.npy")
    
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Error: Could not find image or depth map for ID {IMAGE_ID}")
        return

    print(f"Running inference on {rgb_path}...")
    img = cv2.imread(rgb_path)
    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load ground truth depth map
    depth_map = np.load(depth_path)

    # 2. Run YOLO Inference
    results = model(img_rgb, verbose=False)
    result = results[0]

    # 3. Visualization Setup
    overlay = img.copy()
    
    if result.masks is None or len(result.masks) == 0:
        print("No pallets detected in this image.")
        cv2.imwrite(OUTPUT_PATH, img)
        return

    # Extract original-scaled polygons and bounding boxes
    mask_polygons = result.masks.xy  # <-- FIX: Use exact polygon coordinates
    boxes = result.boxes.xyxy.cpu().numpy()
    
    print(f"Found {len(mask_polygons)} pallets. Processing masks and calculating distances...")

    distances =[]

    # First pass: Draw masks on overlay and calculate depth
    for polygon in mask_polygons:
        # Create a blank black mask exact to image dimensions
        single_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # Draw the YOLO polygon as a solid white shape
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(single_mask, [pts], 255)
        
        # Generate a random color and apply it to the overlay where the mask is white
        color =[int(c) for c in np.random.randint(0, 255, (3,))]
        overlay[single_mask == 255] = color
        
        # Calculate Distance: Get depth values strictly inside this polygon
        pallet_depths = depth_map[single_mask == 255]
        valid_depths = pallet_depths[(pallet_depths > 0.1) & (pallet_depths < 50.0)]
        
        if len(valid_depths) > 0:
            dist_m = np.median(valid_depths)
        else:
            dist_m = 0.0
            
        distances.append(dist_m)

    # Blend original image with the colored masks (50% transparency)
    final_img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # Second pass: Draw bounding boxes and text directly on the blended image
    for box, dist_m in zip(boxes, distances):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw Bounding Box
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw the Text below the box
        text = f"Dist: {dist_m:.2f}m"
        text_x = x1
        text_y = min(img_h - 10, y2 + 25) # Prevent text from going off screen
        
        # Black outline for readability, then white text
        cv2.putText(final_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(final_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 4. Save Output
    cv2.imwrite(OUTPUT_PATH, final_img)
    print(f"Done! Saved visualization to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
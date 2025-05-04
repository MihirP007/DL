# cv_traffic_analyzer.py (Resizable Window, More Vehicle Types, Detection Tuning)

import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys
import os
import traceback # For detailed errors if needed

# --- Configuration ---
IMAGE_SOURCE = "02_fisheye_day_000280.jpg" # Your image file

# Using yolov8m.pt for better accuracy
YOLO_MODEL_PATH = 'yolov8m.pt'
# Or change back to 'yolov8s.pt' or 'yolov8n.pt' if needed

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# --- Manual ROI Definitions ---
# !!! CRITICAL: Ensure these coordinates are accurate for your image !!!
ROIs = {
    "N_approach": np.array([[303, 235], [210, 159], [162, 208], [176, 355], [306, 233]], np.int32),
    "S_approach": np.array([[560, 555], [735, 401], [888, 712], [742, 821], [562, 555]], np.int32),
    "E_approach": np.array([[422, 167], [542, 178], [769, 141], [693, 103], [555, 117]], np.int32),
    "W_approach": np.array([[228, 506], [326, 663], [160, 815],  [116, 782]], np.int32)
}
print("--- Defined ROIs (Estimates - Please Verify/Adjust Coordinates!) ---")
for name, poly in ROIs.items(): print(f"  {name}: {poly.tolist()}")
print("-" * 30)


# --- Vehicle Classes to Count (from YOLO COCO dataset) ---
# <<< --- MODIFIED TO INCLUDE MORE TYPES --- >>>
TARGET_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
TARGET_CLASS_IDS = list(TARGET_CLASSES.keys()) # Update list automatically
# <<< --- END MODIFICATION --- >>>

# --- Helper Function ---
def get_bounding_box_bottom_center(box):
    x1, y1, x2, y2 = map(int, box); center_x = (x1 + x2) // 2; bottom_y = y2
    return (center_x, bottom_y)

# --- Main Processing ---
if __name__ == "__main__":

    if not os.path.exists(IMAGE_SOURCE): print(f"Error: Image file not found: {IMAGE_SOURCE}"); sys.exit(1)

    print(f"Loading YOLOv8 model: {YOLO_MODEL_PATH}...")
    try: model = YOLO(YOLO_MODEL_PATH); print("Model loaded successfully.")
    except Exception as e: print(f"Error loading YOLO model '{YOLO_MODEL_PATH}': {e}"); sys.exit(1)

    print(f"Loading image: {IMAGE_SOURCE}")
    frame = cv2.imread(IMAGE_SOURCE);
    if frame is None: print(f"Error: Could not read image file: {IMAGE_SOURCE}"); sys.exit(1)
    display_frame = frame.copy()

    # --- Create Resizable Window ---
    WINDOW_NAME = "Traffic Analysis Output (Resizable)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_height, frame_width = frame.shape[:2]; initial_width = min(1280, frame_width); initial_height = int(initial_width * (frame_height / frame_width))
    cv2.resizeWindow(WINDOW_NAME, initial_width, initial_height)

    # --- YOLOv8 Inference ---
    start_time = time.time(); print("Running YOLOv8 inference...")
    # Pass the updated list of class IDs to the model's predict function
    results = model.predict(source=frame, classes=TARGET_CLASS_IDS, conf=CONFIDENCE_THRESHOLD, verbose=False)
    inference_time = time.time() - start_time; print(f"Inference completed in {inference_time:.3f} seconds.")

    # --- Vehicle Counting per ROI ---
    # Initialize counts dictionary dynamically based on updated TARGET_CLASSES
    vehicle_counts_per_roi = {roi_name: {'total': 0} for roi_name in ROIs}
    for roi_name in vehicle_counts_per_roi:
        for class_name in TARGET_CLASSES.values(): # Initialize all target types to 0
            vehicle_counts_per_roi[roi_name][class_name] = 0
    detected_vehicles_info = []

    print("\n--- Processing Detections ---")
    if results and results[0].boxes is not None:
        print(f"Detected {len(results[0].boxes)} potential objects (after class/conf filtering).")
        for i, box_data in enumerate(results[0].boxes):
            class_id = int(box_data.cls)
            # Check if the detected class_id is one we are interested in
            if class_id in TARGET_CLASSES:
                vehicle_type = TARGET_CLASSES[class_id]; bbox = box_data.xyxy[0].cpu().numpy()
                confidence = float(box_data.conf); ref_point = get_bounding_box_bottom_center(bbox)
                assigned_roi = None

                print(f"\nChecking Vehicle {i} (Type: {vehicle_type}, Conf: {confidence:.2f}, Ref Point: {ref_point})") # Debug
                for roi_name, roi_polygon in ROIs.items():
                    dist = cv2.pointPolygonTest(roi_polygon, ref_point, False)
                    is_inside = dist >= 0
                    print(f"  - Testing ROI '{roi_name}': pointPolygonTest result = {dist} (Inside={is_inside})") # Debug
                    if is_inside:
                        vehicle_counts_per_roi[roi_name]['total'] += 1
                        # Ensure the type exists in the dict before incrementing
                        if vehicle_type in vehicle_counts_per_roi[roi_name]:
                            vehicle_counts_per_roi[roi_name][vehicle_type] += 1
                        else: # Should not happen with current init, but safe fallback
                             vehicle_counts_per_roi[roi_name][vehicle_type] = 1
                        assigned_roi = roi_name
                        print(f"    >> Assigned to '{roi_name}' << BREAKING inner loop.") # Debug
                        break

                detected_vehicles_info.append({'bbox': bbox, 'point': ref_point, 'type': vehicle_type, 'conf': confidence, 'roi': assigned_roi})

    # --- Display Information ---
    print("\n--- Annotating Frame ---")
    # Draw ROIs and counts (logic remains the same)
    for roi_name, roi_polygon in ROIs.items():
         cv2.polylines(display_frame, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
         count_text = f"{roi_name}: {vehicle_counts_per_roi[roi_name]['total']}"
         text_x, text_y = roi_polygon[0][0], roi_polygon[0][1]; text_y -= 15; text_y = max(text_y, 20); text_x = max(text_x, 10)
         (w, h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2); cv2.rectangle(display_frame, (text_x - 5, text_y - h - 5), (text_x + w + 5, text_y + 5), (0,0,0), -1); cv2.putText(display_frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw detected vehicle boxes and points (logic remains the same)
    for vehicle in detected_vehicles_info:
        x1, y1, x2, y2 = map(int, vehicle['bbox']); point = vehicle['point']
        label = f"{vehicle['type']} ({vehicle['conf']:.2f})"; roi_tag = f"->{vehicle['roi']}" if vehicle['roi'] else ""
        box_color = (0, 0, 255) if vehicle['roi'] else (255, 0, 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
        cv2.putText(display_frame, label + roi_tag, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        point_color = (0, 255, 0) if vehicle['roi'] else (255, 0, 255)
        cv2.circle(display_frame, point, 5, point_color, -1)

    # Display FPS/Inference Time (logic remains the same)
    fps_text = f"FPS: {1.0 / inference_time if inference_time > 0 else 0:.2f} (Inference: {inference_time*1000:.1f}ms)"
    cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- Show/Save Output ---
    output_filename = "annotated_DEBUG_" + os.path.basename(IMAGE_SOURCE)
    print(f"\n--- Saving annotated image to: {output_filename} ---")
    cv2.imwrite(output_filename, display_frame)
    print("Displaying annotated image in resizable window. Press any key to close.")
    cv2.imshow(WINDOW_NAME, display_frame); cv2.waitKey(0)

    # --- Cleanup ---
    cv2.destroyAllWindows()
    print("\n--- Final Counts per ROI ---");
    for roi, counts in vehicle_counts_per_roi.items():
        # <<< Updated print to show breakdown by all tracked types >>>
        type_counts_str = ", ".join([f"{k}:{v}" for k,v in counts.items() if k != 'total' and v > 0])
        print(f"  {roi}: Total={counts['total']} " + (f"({type_counts_str})" if type_counts_str else ""))
        # <<< End Update >>>
    print("-" * 30); print("Processing finished.")
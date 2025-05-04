# define_rois.py
import cv2
import numpy as np
import sys
import os

# --- Configuration ---
IMAGE_PATH = "01_fisheye_day_000378.jpg" # <<< Your intersection image file
WINDOW_NAME = "Define ROIs - Click Corners, Right-Click when done with ROI, Press 'n' for next, 'q' to quit"

# <<< List the exact names of the ROIs you need to define >>>
# These should match the keys you want in the ROIs dictionary later
ROI_NAMES_TO_DEFINE = ["N_approach", "S_approach", "E_approach", "W_approach"]
# Example if you want per-lane ROIs later:
# ROI_NAMES_TO_DEFINE = ["edge_N_in_0", "edge_N_in_1", "edge_S_in_0", ...]

# Global variables to store points and defined ROIs
current_points = []
defined_rois = {}
current_roi_index = 0
original_frame = None
display_frame = None # Frame to draw on

# --- Mouse Callback Function ---
def click_event(event, x, y, flags, param):
    global current_points, display_frame, current_roi_index, defined_rois

    if current_roi_index >= len(ROI_NAMES_TO_DEFINE):
        return # All ROIs defined

    current_roi_name = ROI_NAMES_TO_DEFINE[current_roi_index]

    # On Left Click: Add point and draw feedback
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Added point ({x}, {y}) for ROI '{current_roi_name}'")
        # Draw the point
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1) # Green dot
        # Draw lines connecting points if more than one exists
        if len(current_points) > 1:
            cv2.line(display_frame, current_points[-2], current_points[-1], (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, display_frame)

    # On Right Click: Finalize the current ROI
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) < 3:
            print("WARN: Need at least 3 points to define a polygon ROI. Right-click ignored.")
            return

        # Store the completed ROI
        roi_array = np.array(current_points, np.int32)
        defined_rois[current_roi_name] = roi_array
        print(f"--- ROI '{current_roi_name}' defined with {len(current_points)} points. Press 'n' for next ROI or 'q' to finish. ---")

        # Draw the final polygon permanently (optional)
        cv2.polylines(original_frame, [roi_array], isClosed=True, color=(0, 0, 255), thickness=2) # Draw in Red on original

        # Clear points for the next ROI, reset display frame for next definition
        current_points = []
        #display_frame = original_frame.copy() # Reset drawing overlay - Keep points for now
        # cv2.imshow(WINDOW_NAME, display_frame) # Update view

        # # Automatically advance (optional) - Use 'n' key instead for more control
        # current_roi_index += 1
        # if current_roi_index < len(ROI_NAMES_TO_DEFINE):
        #     print(f"\n>>> Now defining ROI: '{ROI_NAMES_TO_DEFINE[current_roi_index]}' <<<")
        # else:
        #     print("\n--- All ROIs defined! Press 'q' to quit and print results. ---")

# --- Main Loop ---
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'")
        sys.exit(1)

    original_frame = cv2.imread(IMAGE_PATH)
    if original_frame is None:
        print(f"Error: Could not read image file: {IMAGE_PATH}")
        sys.exit(1)
    display_frame = original_frame.copy() # Start with a fresh copy for drawing

    print("\n--- ROI Definition Instructions ---")
    print(f"1. Window shows: {IMAGE_PATH}")
    print(f"2. Currently defining ROI: '{ROI_NAMES_TO_DEFINE[current_roi_index]}'")
    print("3. LEFT-CLICK on the image to add corner points for this ROI in order.")
    print("4. RIGHT-CLICK when you have added all points for the current ROI.")
    print("5. Press 'n' AFTER Right-Clicking to move to the NEXT ROI.")
    print("6. Press 'q' anytime to FINISH and print the defined ROI coordinates.")
    print("-" * 30)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, click_event)

    while True:
        # Display current instructions on the image
        instruction_frame = display_frame.copy() # Copy frame to add text overlay
        if current_roi_index < len(ROI_NAMES_TO_DEFINE):
            text = f"Define: {ROI_NAMES_TO_DEFINE[current_roi_index]}. Left-Click corners. Right-Click when done."
        else:
            text = "All ROIs Defined! Press 'q' to finish."
        cv2.putText(instruction_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) # Black outline
        cv2.putText(instruction_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text

        cv2.imshow(WINDOW_NAME, instruction_frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break
        # Next ROI (only after Right-Clicking to confirm current one)
        elif key == ord('n'):
            if not current_points: # Allow 'n' only if current points list is empty (meaning last ROI was finalized)
                current_roi_index += 1
                if current_roi_index < len(ROI_NAMES_TO_DEFINE):
                    print(f"\n>>> Switched to defining ROI: '{ROI_NAMES_TO_DEFINE[current_roi_index]}' <<<")
                    # Reset display frame to show only finalized ROIs for clarity
                    display_frame = original_frame.copy()
                    # Re-draw already defined ROIs
                    for name, poly in defined_rois.items():
                         cv2.polylines(display_frame, [poly], isClosed=True, color=(0,0,255), thickness=2)
                else:
                    print("\n--- All ROIs defined! Press 'q' to quit and print results. ---")
                    # Keep loop running until user presses 'q'
            else:
                print("Please Right-Click first to finalize the current ROI points before pressing 'n'.")

    cv2.destroyAllWindows()

    # --- Print the results in the required format ---
    print("\n\n--- Defined ROI Coordinates ---")
    print("# Copy and paste this dictionary into your cv_traffic_analyzer.py script")
    print("ROIs = {")
    for i, (name, points_array) in enumerate(defined_rois.items()):
        # Convert to list of lists for printing
        points_list = points_array.tolist()
        print(f"    \"{name}\": np.array({points_list}, np.int32)", end="")
        if i < len(defined_rois) - 1:
            print(",") # Add comma if not the last item
        else:
            print("") # Newline for the last item
    print("}")
    print("-" * 30)
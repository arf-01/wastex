import cv2
import time
import os
import logging
from datetime import datetime

# --- CONFIGURATION ---
CAPTURE_DIR = "/home/dhruba001/webcam_captures"
CAMERA_INDEX = 0

PROCESS_RESOLUTION = (320, 240)
LOOP_DELAY_SEC = 0.1

# 1. Region of Interest (ROI) - Ignore the top and sides where people walk
# Using (ymin:ymax, xmin:xmax). Assuming 320x240 frame, crop to center/bottom
ROI_Y = (20, 240)  # Cut off top 20 pixels
ROI_X = (40, 280)  # Cut off 40 pixels from left and right edges

# 2 & 4. Dual Thresholds (Hysteresis)
DIFF_THRESHOLD = 25              
DETECT_HIGH_AREA = 800          # High threshold: Triggers an object placement
EMPTY_LOW_AREA = 300             # Low threshold: Confirms pan is empty

# 3. Temporal Consistency
FRAMES_TO_DETECT = 5             # Must see object for 5 frames straight
FRAMES_TO_EMPTY = 10             # Must be empty for 10 frames straight
STABLE_AREA_DIFF = 1000          # Wait for it to stop wobbling (area difference)
STABLE_FRAMES_REQUIRED = 15      # Frames required before triggering the capture

# 5. Adaptive Baseline (Running Average)
ALPHA = 0.05                     # Speed of baseline update. 0.05 is a slow, steady fade.

os.makedirs(CAPTURE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_frame(frame):
    """Resize, Crop ROI, grayscale, blur, and extract EDGES (Lighting invariant!)"""
    resized = cv2.resize(frame, PROCESS_RESOLUTION)
    roi = resized[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection natively ignores sweeping exposure/lighting changes
    edges = cv2.Canny(blurred, 30, 130)
    return edges

def get_max_contour_area(contours):
    if not contours:
        return 0
    return max([cv2.contourArea(c) for c in contours])

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Failed to open camera on index {CAMERA_INDEX}.")
        return

    logger.info("Starting camera warmup...")
    time.sleep(2.0)
    for _ in range(5): cap.grab()
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read from camera. Exiting...")
        return
        
    # Baseline must be float32 for accumulateWeighted
    baseline = process_frame(frame).astype("float") 
    
    # Initialize Thesis Video Writer
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = os.path.join(CAPTURE_DIR, "demo_session.avi")
    out = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))
    logger.info(f"Adaptive Baseline initialized. Demo Video Recording to: {out_path} ...")

    # STATE MACHINE REPLACED WITH BOOLEAN FLAG (User's Idea!)
    object_already_captured = False
    capture_flash_timer = 0
    motion_count = 0
    empty_count = 0
    stable_count = 0
    previous_area = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to retrieve frame.")
                time.sleep(1)
                continue

            processed_frame = process_frame(frame)
            
            # Simple absdiff using the dynamically running average baseline
            frame_delta = cv2.absdiff(cv2.convertScaleAbs(baseline), processed_frame)
            _, thresh = cv2.threshold(frame_delta, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Dilate aggressively (6 iterations) so Canny edge outlines bleed into a solid filled blob
            thresh = cv2.dilate(thresh, None, iterations=6)
            
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_max_area = get_max_contour_area(contours)

            # --- USER'S BOOLEAN FLAG LOGIC ---

            # BRANCH 1: Pan has cleared (Area is very small)
            if current_max_area < EMPTY_LOW_AREA:
                empty_count += 1
                if empty_count >= FRAMES_TO_EMPTY:
                    if object_already_captured:
                        logger.info("Pan is clear! Resetting flag to 0.")
                        object_already_captured = False
                    
                    # 5. Adaptive background magic (only runs when empty!)
                    cv2.accumulateWeighted(processed_frame, baseline, ALPHA)
                    motion_count = 0
                    stable_count = 0
                    empty_count = 0
                    
            # BRANCH 2: An object is placed (Area is large) OR we are already tracking it
            elif current_max_area > DETECT_HIGH_AREA or motion_count > 0:
                empty_count = 0
                
                # Have we already taken a photo of this object?
                if object_already_captured == False:
                    motion_count += 1
                    
                    if motion_count == 10:
                        logger.info(f"Item detected! Area: {current_max_area}. Waiting 4 seconds for it to settle...")
                        
                    elif motion_count == 25:
                        logger.info(f"Target verifying... Area: {current_max_area}")
                        
                    # Wait exactly 40 frames (4.0 seconds) to guarantee the hand is gone and item settled
                    elif motion_count >= 40:
                        logger.info(f"Item settled perfectly. Capturing...")
                        
                        # Flush buffer for crisp capture
                        for _ in range(5): cap.grab()
                        ret, hq_frame = cap.read()
                        if ret:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"capture_{timestamp}.jpg"
                            filepath = os.path.join(CAPTURE_DIR, filename)
                            cv2.imwrite(filepath, hq_frame)
                            logger.info(f"Image successfully saved: {filepath}")
                        
                        # SET THE BOOLEAN FLAG TO 1 SO WE DON'T CAPTURE AGAIN
                        object_already_captured = True
                        capture_flash_timer = 20  # Flash "CAPTURED" for 2 seconds
                        motion_count = 0
                else:
                    # We already captured it! Do nothing until area drops back to 0.
                    pass

            # Always track previous area at the very end of the loop, for ALL branches!
            previous_area = current_max_area

            # --- DRAW THESIS OVERLAY ---
            # Create a 3-channel (color) miniature version of the Canny edges
            pip_w, pip_h = int(w * 0.3), int(h * 0.3)
            preview = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            preview = cv2.resize(preview, (pip_w, pip_h))
            
            # Place the miniature in the top right corner
            frame[0:pip_h, w-pip_w:w] = preview
            
            # Burn HUD Text into the frame
            if capture_flash_timer > 0:
                hud_text = "*** IMAGE CAPTURED ***"
                color = (0, 255, 255)  # Bright Yellow
                capture_flash_timer -= 1
            elif object_already_captured:
                hud_text = "PLEASE REMOVE ITEM"
                color = (0, 0, 255)    # Red
            else:
                hud_text = "WAITING FOR ITEM"
                color = (0, 255, 0)    # Green
                
            cv2.putText(frame, f"STATUS: {hud_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"AREA: {current_max_area}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Canny Engine", (w - pip_w, pip_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Save frame to the running .avi video
            if out is not None:
                out.write(frame)

            time.sleep(LOOP_DELAY_SEC)

    except KeyboardInterrupt:
        logger.info("Shutting down manually via keyboard interrupt.")
    except Exception as e:
        logger.error(f"Unexpected crash: {e}")
    finally:
        cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        logger.info("Camera & VideoWriter released. Exiting.")

if __name__ == "__main__":
    main()

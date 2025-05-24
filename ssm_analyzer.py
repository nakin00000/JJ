# This is the main script for the SSM (Surrogate Safety Measures) analyzer.
# It processes drone video footage to detect vehicles, track them, and calculate
# various safety metrics like Time-to-Collision (TTC), Post-Encroachment Time (PET),
# and Gap Time. Results are saved to a CSV file, and an annotated video can optionally be generated.

# --- Configuration Constants ---
# These constants define default behaviors and thresholds for the analysis.
# Some can be overridden by command-line arguments.

# PIXELS_PER_METER: Scale factor for converting pixel distances to meters.
# This is a CRITICAL CALIBRATION value and depends heavily on the camera, lens,
# and drone altitude for a specific video. It should be accurately determined.
# Can be overridden by the --pixels_per_meter command-line argument.
PIXELS_PER_METER = 10.0

# DRONE_ALTITUDE_METERS: The altitude of the drone in meters.
# Currently informational; not directly used in calculations unless a more complex
# calibration model is implemented that requires altitude.
DRONE_ALTITUDE_METERS = 90.0 

# PET_HISTORY_FRAMES: Number of past frames to search for prior overlaps when calculating PET.
# E.g., 60 frames at 30 FPS corresponds to a 2-second history.
PET_HISTORY_FRAMES = 60 

# LATERAL_THRESHOLD_METERS: Maximum lateral distance (in meters, typically along X-axis)
# between two vehicles for them to be considered in the same "lane" or path
# for Gap Time calculation.
LATERAL_THRESHOLD_METERS = 2.0 

# VELOCITY_ESTIMATION_WINDOW_FRAMES: Maximum number of frames to look back
# from the current frame to find a previous point for velocity estimation.
# A smaller window reacts faster to speed changes but can be noisier.
VELOCITY_ESTIMATION_WINDOW_FRAMES = 5 

# CRITICAL_TTC_THRESHOLD: Time-to-Collision (TTC) value in seconds below which
# a TTC is considered critical and will be highlighted in annotations.
CRITICAL_TTC_THRESHOLD = 5.0 

# Note: VIDEO_FPS is determined from the input video itself during the call to load_video().

import cv2
import os
import numpy as np 
import math # For float('inf'), sqrt
import csv 
import argparse 
import torch # PyTorch is a core dependency for device management and ML models

# --- Conditional Imports for Core ML/DL Libraries ---
# These flags indicate whether the necessary libraries for object detection (YOLO)
# and tracking (DeepSORT) were successfully imported.
TORCH_INSTALLED = True # Assume torch imported successfully, or it would have errored at the global import.
ULTRALYTICS_INSTALLED = False # Flag for Ultralytics YOLO library
DEEPSORT_INSTALLED = False  # Flag for DeepSORT library

try:
    from ultralytics import YOLO 
    ULTRALYTICS_INSTALLED = True
except ImportError:
    print("Info: Ultralytics (YOLO) library not found. "
          "YOLO-based object detection will be skipped if models are attempted to load.")

try:
    # DeepSORT for object tracking. The specific import path might vary based on the chosen DeepSORT fork.
    # This example assumes a common structure for deep_sort_pytorch.
    from deep_sort_pytorch.utils.parser import get_config
    from deep_sort_pytorch.deep_sort import DeepSort
    if TORCH_INSTALLED: # DeepSORT typically relies on PyTorch for its ReID model.
        DEEPSORT_INSTALLED = True
    else:
        print("Info: DeepSORT cannot be enabled because PyTorch (torch) is not installed.")
except ImportError:
    print("Info: deep_sort_pytorch library not found or not in PYTHONPATH. "
          "DeepSORT-based object tracking will be skipped if models are attempted to load.")


def load_video(video_path):
    """
    Loads a video from the specified path and extracts its properties.

    Args:
        video_path (str): The file path to the video.

    Returns:
        tuple: A tuple containing:
            - cap (cv2.VideoCapture or None): The OpenCV video capture object if successful, None otherwise.
            - fps (float): Frames per second of the video. Returns a default (e.g., 30.0) if FPS is unreadable or 0.
            - frame_width (int): Width of the video frames in pixels.
            - frame_height (int): Height of the video frames in pixels.
        Returns (None, 0, 0, 0) if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, 0, 0, 0 
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        print("Warning: Video FPS reported as 0. Using default 30.0 for calculations.")
        fps = 30.0 # Fallback FPS if video metadata is problematic
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, frame_width, frame_height

def detect_vehicles_yolo(frame, model):
    """
    Detects vehicles in a given frame using a pre-loaded YOLO model.
    The model object passed to this function might be running on CPU or CUDA.
    The Ultralytics library handles device-specific operations internally for inference.

    Args:
        frame (numpy.ndarray): The input video frame (OpenCV BGR format).
        model (ultralytics.YOLO): The pre-loaded YOLO model object.

    Returns:
        list: A list of detections. Each detection is a list of the format
              [x1, y1, x2, y2, confidence, class_id], where (x1, y1) and (x2, y2)
              are the top-left and bottom-right pixel coordinates of the bounding box.
              Returns an empty list if YOLO library/model is not available or no vehicles are detected.
    """
    if not ULTRALYTICS_INSTALLED or not model: 
        return [] 
        
    detections = []
    # Perform inference. `verbose=False` suppresses detailed YOLO console output.
    # The `model(frame)` call handles data transfer to the model's device (CPU/CUDA)
    # and returns results, typically as CPU tensors.
    results = model(frame, verbose=False) 
    
    # COCO class IDs for common vehicles. Adjust if using a model with different classes.
    vehicle_class_ids = [2, 3, 5, 7] # car, motorcycle, bus, truck in COCO

    if results and results[0].boxes:
        # results[0].boxes.data usually contains a tensor with [x1, y1, x2, y2, conf, cls]
        for box_data in results[0].boxes.data: 
            x1, y1, x2, y2, confidence, class_id = box_data.tolist() # Convert tensor row to list
            if int(class_id) in vehicle_class_ids:
                detections.append([x1, y1, x2, y2, confidence, int(class_id)])
    return detections

def initialize_deepsort_tracker(model_path="osnet_x0_25_msmt17.pt"):
    """
    Initializes a PyTorch-compatible DeepSORT tracker, attempting CUDA usage if available.
    This function assumes a DeepSORT library structure similar to `deep_sort_pytorch`.
    The user is responsible for ensuring the chosen DeepSORT library, its configuration,
    and the ReID model are compatible and correctly set up.

    Args:
        model_path (str): Path or name of the Re-identification (ReID) model checkpoint file
                          (e.g., "osnet_x0_25_msmt17.pt"). The specific file and its
                          expected location depend on the chosen DeepSORT library.

    Returns:
        DeepSort object or None: The initialized DeepSORT tracker object if successful, None otherwise.
    """
    if not DEEPSORT_INSTALLED: 
        print("Info: DeepSORT library not available or not enabled; skipping tracker initialization.")
        return None
    
    cfg = None # Initialize cfg to None
    try:
        # Attempt to load configuration using get_config() from the DeepSORT library.
        # This configuration often dictates ReID model parameters, NMS settings, etc.
        cfg = get_config() 
    except Exception as e:
        print(f"Warning: Could not load DeepSORT config using get_config(): {e}. "
              "The tracker might use hardcoded defaults or fail if a valid config is essential. "
              "Ensure your DeepSORT library is correctly installed and configured.")
        # Depending on the DeepSORT library, a missing or partial config might be problematic.
        # For this implementation, we proceed and let DeepSort constructor handle it,
        # using example defaults below if cfg attributes are missing.

    # Determine if CUDA should be used for DeepSORT (ReID model typically)
    use_cuda_for_deepsort = torch.cuda.is_available() and TORCH_INSTALLED
    if use_cuda_for_deepsort:
        print("Info: DeepSORT tracker will attempt to use CUDA for its ReID model.")
    else:
        print("Info: CUDA not available or PyTorch not installed. DeepSORT will run on CPU.")

    try:
        # These are common DeepSORT parameters. Their exact names and whether they are
        # loaded from `cfg` or passed directly depends on the specific DeepSORT library version.
        # The defaults provided here are examples and might need adjustment.
        ds_max_dist = cfg.DEEPSORT.MAX_DIST if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'MAX_DIST') else 0.2 
        ds_min_confidence = cfg.DEEPSORT.MIN_CONFIDENCE if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'MIN_CONFIDENCE') else 0.3
        ds_nms_max_overlap = cfg.DEEPSORT.NMS_MAX_OVERLAP if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'NMS_MAX_OVERLAP') else 1.0
        ds_max_iou_distance = cfg.DEEPSORT.MAX_IOU_DISTANCE if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'MAX_IOU_DISTANCE') else 0.7
        ds_max_age = cfg.DEEPSORT.MAX_AGE if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'MAX_AGE') else 70
        ds_n_init = cfg.DEEPSORT.N_INIT if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'N_INIT') else 3
        ds_nn_budget = cfg.DEEPSORT.NN_BUDGET if cfg and hasattr(cfg, 'DEEPSORT') and hasattr(cfg.DEEPSORT, 'NN_BUDGET') else 100

        tracker = DeepSort( 
            model_path=model_path, # Path to the ReID model checkpoint
            max_dist=ds_max_dist, 
            min_confidence=ds_min_confidence,
            nms_max_overlap=ds_nms_max_overlap, 
            max_iou_distance=ds_max_iou_distance,
            max_age=ds_max_age, 
            n_init=ds_n_init, 
            nn_budget=ds_nn_budget, 
            use_cuda=use_cuda_for_deepsort # Flag to indicate CUDA usage preference
        )
        return tracker
    except Exception as e:
        print(f"Error initializing DeepSORT tracker: {e}")
        print("Ensure the ReID model weights (e.g., osnet_x0_25_msmt17.pt from --reid_model_path) "
              "and DeepSORT configurations are compatible and correctly placed for your chosen DeepSORT library.")
        return None

def track_vehicles_deepsort(frame, detections, tracker):
    """
    Tracks detected vehicles using the DeepSORT algorithm.
    Assumes detections are CPU-based lists/NumPy arrays, and the frame is an OpenCV (NumPy) frame.
    The DeepSORT library handles internal device management for its ReID model if CUDA is enabled.

    Args:
        frame (numpy.ndarray): The current video frame (OpenCV BGR format).
        detections (list): A list of detections from YOLO, where each detection is
                           [x1, y1, x2, y2, confidence, class_id].
        tracker (DeepSort object): The initialized DeepSORT tracker.

    Returns:
        list: A list of tracked objects. Each object is [x1, y1, x2, y2, track_id (str)].
              Returns an empty list if issues occur or no objects are tracked.
    """
    if not DEEPSORT_INSTALLED or not tracker: 
        return [] 
    
    if not detections: 
        # Update tracker even with no detections for its internal state management (e.g., track aging).
        # DeepSORT might require an RGB frame.
        empty_frame_for_update = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
        if empty_frame_for_update is not None:
            # Pass empty NumPy arrays for detection data.
            tracker.update(np.array([]), np.array([]), np.array([]), empty_frame_for_update)
        return []

    xywhs = []; confs = []; oids = [] # DeepSORT expects these lists
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        # Convert [x1,y1,x2,y2] to [center_x, center_y, width, height]
        xywhs.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confs.append(conf)
        oids.append(class_id) 
    
    # DeepSORT's ReID model typically expects an RGB frame.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    # Convert lists to NumPy arrays as expected by many DeepSORT implementations.
    np_xywhs = np.array(xywhs, dtype=float)
    np_confs = np.array(confs, dtype=float)
    np_oids = np.array(oids, dtype=int) # Assuming class IDs are integers

    # Call the tracker's update method.
    tracker.update(np_xywhs, np_confs, np_oids, rgb_frame)
    
    tracked_output = []
    # Process tracker.outputs to get final track data.
    # Format of tracker.outputs can vary slightly by DeepSORT library version.
    # Typically: [x1, y1, x2, y2, track_id, class_id (optional), conf (optional)]
    for track_data in tracker.outputs:
        if len(track_data) >= 5: # Check for minimum expected data
            x1, y1, x2, y2, track_id = track_data[0:5]
            # Ensure coordinates are integers and track_id is a string for consistency.
            tracked_output.append([int(x1), int(y1), int(x2), int(y2), str(track_id)])
    return tracked_output

def calibrate_pixel_to_meters(pixel_x, pixel_y, current_pixels_per_meter):
    """Converts pixel coordinates to world coordinates (in meters) using a simple scale factor."""
    if current_pixels_per_meter <= 0: 
        raise ValueError("pixels_per_meter_scale must be positive.")
    # Simple linear scaling. Assumes origin (0,0) in pixel space maps to (0,0) in world space.
    # Y-axis direction in world space might need inversion (e.g., frame_height_pixels - pixel_y)
    # depending on the desired world coordinate system setup, not handled here.
    world_x = pixel_x / current_pixels_per_meter
    world_y = pixel_y / current_pixels_per_meter 
    return world_x, world_y

def update_vehicle_trajectories(current_tracked_objects, frame_id, current_pixels_per_meter, current_trajectories_dict):
    """
    Updates a given dictionary of vehicle trajectories with new tracking data for the current frame.
    Each trajectory point stores frame ID, world coordinates (calibrated from bottom-center 
    of pixel bbox), and the full pixel bounding box.

    Args:
        current_tracked_objects (list): List of objects tracked in the current frame.
                                        Each item: [x1_pix, y1_pix, x2_pix, y2_pix, track_id].
        frame_id (int): The ID of the current frame.
        current_pixels_per_meter (float): The current pixel-to-meter scale factor.
        current_trajectories_dict (dict): The dictionary to update with trajectory data.
            Format: {track_id: [(frame_id, world_x, world_y, px1, py1, px2, py2), ...], ...}
            This dictionary is modified in-place.
    """
    for obj_data in current_tracked_objects:
        x1_pix, y1_pix, x2_pix, y2_pix, track_id_str = obj_data
        
        # Use bottom-center of the pixel bounding box for world coordinate mapping.
        # This point is often more stable for vehicles viewed from above.
        pixel_center_x_for_world = (x1_pix + x2_pix) / 2.0
        pixel_center_y_for_world = float(y2_pix) # Bottom-center y-coordinate
        
        try:
            world_x, world_y = calibrate_pixel_to_meters( 
                pixel_center_x_for_world, 
                pixel_center_y_for_world, 
                current_pixels_per_meter 
            )
            
            track_id_key = str(track_id_str) # Ensure track_id is string for dict key consistency

            if track_id_key not in current_trajectories_dict:
                current_trajectories_dict[track_id_key] = [] # Initialize list for new track
            
            # Append new point: (frame_id, world_x, world_y, and original pixel bbox)
            current_trajectories_dict[track_id_key].append(
                (frame_id, world_x, world_y, 
                 float(x1_pix), float(y1_pix), float(x2_pix), float(y2_pix)) 
            )
        except ValueError as e: # Catch calibration errors (e.g., invalid pixels_per_meter)
            print(f"Warning: Calibration error for track_id {track_id_str}, frame {frame_id}: {e}")
        except Exception as e: # Catch any other unexpected errors during this update
            print(f"Warning: Trajectory update error for track_id {track_id_str}, frame {frame_id}: {e}")

def _get_trajectory_point(trajectory_points, frame_id_to_find):
    """
    Helper function to find a specific data point in a vehicle's trajectory list by its frame_id.
    Searches backwards from the end of the list, as recent points are often needed.

    Args:
        trajectory_points (list): A list of tuples, where each tuple is a trajectory point:
                                  (frame_id, world_x, world_y, px1, py1, px2, py2).
        frame_id_to_find (int): The frame ID of the point to retrieve.

    Returns:
        tuple or None: The trajectory point tuple if found, otherwise None.
    """
    for point in reversed(trajectory_points): # Iterate backwards for efficiency if recent points are common
        if point[0] == frame_id_to_find:
            return point
    return None

def _get_vehicle_velocity(trajectory_full, current_frame_id, fps, 
                           velocity_estimation_window_frames=VELOCITY_ESTIMATION_WINDOW_FRAMES):
    """
    Estimates a vehicle's velocity (vx, vy) in meters per second using its trajectory.
    Velocity is calculated based on the change in world coordinates between the current
    point and the most recent previous point found within the estimation window.

    Args:
        trajectory_full (list): The vehicle's complete list of trajectory points.
        current_frame_id (int): The frame ID for which to estimate velocity (current point).
        fps (float): Frames per second of the video.
        velocity_estimation_window_frames (int): Max number of past frames to search for a
                                                 previous point for velocity calculation.

    Returns:
        tuple (vx, vy) or (None, None): Estimated velocity components (vx, vy) in m/s,
                                        or (None, None) if velocity cannot be determined
                                        (e.g., insufficient trajectory history within the window).
    """
    cp = _get_trajectory_point(trajectory_full, current_frame_id) # Current point
    if not cp: 
        return None, None # No data for current frame

    pp = None # Previous point
    # Search backwards from current_frame_id - 1 up to window limit
    for i in range(1, velocity_estimation_window_frames + 1):
        prev_frame_id = current_frame_id - i
        if prev_frame_id < 0: # Reached before the start of the video
            break
        pp_candidate = _get_trajectory_point(trajectory_full, prev_frame_id)
        if pp_candidate:
            pp = pp_candidate
            break # Found the most recent valid previous point
            
    if not pp: 
        return None, None # No suitable previous point found

    # Extract world coordinates (indices 1 and 2) from the point tuples
    x_curr, y_curr = cp[1], cp[2]
    x_prev, y_prev = pp[1], pp[2]
    
    frame_diff = cp[0] - pp[0] # Difference in frame numbers (should be positive)
    if frame_diff <= 0: 
        # This case implies points are not distinct in time or ordered incorrectly.
        return None, None 
    
    time_diff_seconds = frame_diff / fps
    vx = (x_curr - x_prev) / time_diff_seconds
    vy = (y_curr - y_prev) / time_diff_seconds
    return vx, vy

def check_bounding_box_overlap(box1, box2):
    """
    Checks if two bounding boxes (x1, y1, x2, y2) overlap.

    Args:
        box1 (tuple): Coordinates (x1, y1, x2, y2) of the first box.
        box2 (tuple): Coordinates (x1, y1, x2, y2) of the second box.
                      Assumes x2 >= x1 and y2 >= y1 for both boxes.

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    # Check for non-overlap conditions
    # True if box1 is to the left of box2, or box1 is to the right of box2, etc.
    if box1_x2 < box2_x1 or box1_x1 > box2_x2 or \
       box1_y2 < box2_y1 or box1_y1 > box2_y2:
        return False
    return True # If none of the non-overlap conditions are met, they overlap

def calculate_ttc_for_pair(traj1_full, traj2_full, current_frame_id, fps):
    """
    Calculates Time-to-Collision (TTC) in seconds between two vehicles.
    Assumes a constant velocity model for prediction based on recent trajectory.

    Args:
        traj1_full (list): Full trajectory data for vehicle 1.
        traj2_full (list): Full trajectory data for vehicle 2.
        current_frame_id (int): The current frame ID for calculation.
        fps (float): Video frames per second.

    Returns:
        float: Calculated TTC in seconds. Returns float('inf') if no collision is
               predicted (e.g., vehicles moving apart, parallel, or data insufficient for velocity).
    """
    cp1 = _get_trajectory_point(traj1_full, current_frame_id)
    cp2 = _get_trajectory_point(traj2_full, current_frame_id)
    if not cp1 or not cp2: return float('inf')

    vx1, vy1 = _get_vehicle_velocity(traj1_full, current_frame_id, fps)
    vx2, vy2 = _get_vehicle_velocity(traj2_full, current_frame_id, fps)

    if vx1 is None or vx2 is None: return float('inf') # Velocity couldn't be estimated
    
    x1_curr, y1_curr = cp1[1], cp1[2] # World coordinates
    x2_curr, y2_curr = cp2[1], cp2[2]
    
    dx = x1_curr - x2_curr; dy = y1_curr - y2_curr       # Relative position vector
    dvx = vx1 - vx2; dvy = vy1 - vy2                     # Relative velocity vector
    
    # Dot product of relative position and relative velocity
    # If >= 0, vehicles are not on a collision course (moving apart or parallel in a non-closing way)
    dot_product_rel_pos_vel = dvx * dx + dvy * dy
    if dot_product_rel_pos_vel >= 0: 
        return float('inf')
    
    distance_squared = dx**2 + dy**2
    if distance_squared == 0: # Already at the same point
        return 0.0 
    
    # TTC formula: ratio of squared distance to the negative of the dot product
    # (since dot_product_rel_pos_vel is negative if they are closing)
    ttc = distance_squared / -dot_product_rel_pos_vel

    return ttc if ttc >= 0 else float('inf') # Ensure TTC is not negative (past collision)

def calculate_all_ssms(current_vehicle_trajectories, current_frame_id, fps):
    """
    Calculates all selected Surrogate Safety Measures (SSMs) for relevant vehicle pairs
    in the current frame. Currently includes TTC, PET, and Gap Time.

    Args:
        current_vehicle_trajectories (dict): All known vehicle trajectories.
            Format: {track_id: [(frame_id, world_x, world_y, px1, py1, px2, py2), ...]}
        current_frame_id (int): The frame ID for which SSMs are being calculated.
        fps (float): Video frames per second.

    Returns:
        dict: SSM results for vehicles involved in interactions in the current frame.
            Format: {(frame_id, track_id): {'TTC': value, 'PET': value, 'GapTime': value}, ...}
            SSM values are in seconds, or float('inf') if not applicable/calculable.
    """
    ssm_results = {} 
    # Filter for vehicles that have a trajectory point in the current frame
    active_track_ids = [ track_id for track_id, trajectory in current_vehicle_trajectories.items() 
                         if _get_trajectory_point(trajectory, current_frame_id) ]

    # Iterate through unique pairs of active vehicles
    for i in range(len(active_track_ids)):
        for j in range(i + 1, len(active_track_ids)):
            id1, id2 = active_track_ids[i], active_track_ids[j]
            traj1_full = current_vehicle_trajectories[id1]
            traj2_full = current_vehicle_trajectories[id2]

            # Initialize SSM dictionary entries for both vehicles for this frame
            key1 = (current_frame_id, id1); key2 = (current_frame_id, id2)
            for key in [key1, key2]:
                if key not in ssm_results: 
                    ssm_results[key] = {'TTC': float('inf'), 'PET': float('inf'), 'GapTime': float('inf')}

            # --- TTC Calculation ---
            current_ttc = calculate_ttc_for_pair(traj1_full, traj2_full, current_frame_id, fps)
            # Update TTC for both vehicles (minimum TTC if multiple interactions)
            ssm_results[key1]['TTC'] = min(ssm_results[key1]['TTC'], current_ttc)
            ssm_results[key2]['TTC'] = min(ssm_results[key2]['TTC'], current_ttc)

            # --- PET Calculation ---
            current_pet_for_pair = float('inf')
            point1_curr = _get_trajectory_point(traj1_full, current_frame_id)
            point2_curr = _get_trajectory_point(traj2_full, current_frame_id)
            
            if point1_curr and point2_curr: # Both vehicles must exist in current frame
                # Pixel BBoxes are at indices 3,4,5,6 of the trajectory point tuple
                box1_curr = (point1_curr[3], point1_curr[4], point1_curr[5], point1_curr[6])
                box2_curr = (point2_curr[3], point2_curr[4], point2_curr[5], point2_curr[6])

                # PET is calculated if vehicles are NOT currently overlapping
                if not check_bounding_box_overlap(box1_curr, box2_curr):
                    # Search backwards for the most recent frame where they DID overlap
                    for k_hist in range(1, PET_HISTORY_FRAMES + 1):
                        hist_frame_id = current_frame_id - k_hist
                        if hist_frame_id < 0: break # Reached start of video history

                        p1_hist = _get_trajectory_point(traj1_full, hist_frame_id)
                        p2_hist = _get_trajectory_point(traj2_full, hist_frame_id)

                        if p1_hist and p2_hist: # Both vehicles existed at this historical frame
                            b1_hist = (p1_hist[3], p1_hist[4], p1_hist[5], p1_hist[6])
                            b2_hist = (p2_hist[3], p2_hist[4], p2_hist[5], p2_hist[6])
                            
                            if check_bounding_box_overlap(b1_hist, b2_hist):
                                # Found the last frame of overlap. PET is time since then.
                                current_pet_for_pair = (current_frame_id - hist_frame_id) / fps
                                break # Stop search, found most recent encroachment
            
            if current_pet_for_pair != float('inf'):
                 ssm_results[key1]['PET'] = min(ssm_results[key1]['PET'], current_pet_for_pair)
                 ssm_results[key2]['PET'] = min(ssm_results[key2]['PET'], current_pet_for_pair)
            
            # --- Gap Time (Time Headway) Calculation ---
            if point1_curr and point2_curr: # Re-use current points
                world_x1, world_y1 = point1_curr[1], point1_curr[2] # World coords are at index 1,2
                world_x2, world_y2 = point2_curr[1], point2_curr[2]

                vx1, vy1 = _get_vehicle_velocity(traj1_full, current_frame_id, fps)
                vx2, vy2 = _get_vehicle_velocity(traj2_full, current_frame_id, fps)

                speed1 = math.sqrt(vx1**2 + vy1**2) if vx1 is not None and vy1 is not None else 0.0
                speed2 = math.sqrt(vx2**2 + vy2**2) if vx2 is not None and vy2 is not None else 0.0
                
                follower_id_for_gap = None; distance_gap = float('inf'); follower_speed_for_gap = 0.0
                
                # Check for lateral alignment (simplified to X-axis proximity)
                if abs(world_x1 - world_x2) < LATERAL_THRESHOLD_METERS:
                    # Determine leader/follower based on Y-axis (assuming Y is primary direction of motion)
                    if world_y1 > world_y2: # Vehicle 1 is leader, Vehicle 2 is follower
                        distance_gap = world_y1 - world_y2 # Longitudinal distance
                        follower_speed_for_gap = speed2
                        follower_id_for_gap = id2
                    elif world_y2 > world_y1: # Vehicle 2 is leader, Vehicle 1 is follower
                        distance_gap = world_y2 - world_y1
                        follower_speed_for_gap = speed1
                        follower_id_for_gap = id1
                
                if follower_id_for_gap and distance_gap != float('inf'):
                    gap_time_value = float('inf')
                    # Calculate GapTime if follower is moving (speed > 0.1 m/s threshold to avoid division by zero/small numbers)
                    if follower_speed_for_gap > 0.1: 
                        gap_time_value = distance_gap / follower_speed_for_gap
                    
                    key_follower = (current_frame_id, follower_id_for_gap)
                    # ssm_results[key_follower] should exist due to earlier initialization for this frame_id and track_id
                    ssm_results[key_follower]['GapTime'] = min(ssm_results[key_follower]['GapTime'], gap_time_value)
    return ssm_results

def save_results_to_csv(all_trajectories, all_ssm_results, output_csv_path):
    """
    Saves all trajectory data and associated SSM results to a CSV file.
    Each row in the CSV represents a vehicle's state at a specific frame,
    including its world and pixel coordinates, and any calculated SSMs for that vehicle
    at that frame (which are typically interactions with other vehicles).

    Args:
        all_trajectories (dict): The master dictionary of all vehicle trajectories.
            Format: {track_id: [(frame_id, wx, wy, px1, py1, px2, py2), ...], ...}
        all_ssm_results (dict): The master dictionary of all SSM results.
            Format: {(frame_id, track_id): {'TTC': val, 'PET': val, 'GapTime': val}, ...}
        output_csv_path (str): Path for the output CSV file.
    """
    header = ['frame_id', 'vehicle_id', 'world_x', 'world_y', 
              'pixel_x1', 'pixel_y1', 'pixel_x2', 'pixel_y2', 
              'TTC', 'PET', 'GapTime']
    
    rows_to_write = []
    # Iterate through each vehicle's trajectory
    for track_id, trajectory_points in all_trajectories.items():
        # For each point (frame) in that vehicle's trajectory
        for point_data in trajectory_points:
            frame_id, wx, wy, px1, py1, px2, py2 = point_data
            
            # Get SSM values for this specific vehicle at this specific frame
            # The key for all_ssm_results is (frame_id, track_id)
            ssm_values_for_vehicle_at_frame = all_ssm_results.get((frame_id, track_id), {})
            ttc = ssm_values_for_vehicle_at_frame.get('TTC', float('inf'))
            pet = ssm_values_for_vehicle_at_frame.get('PET', float('inf'))
            gap_time = ssm_values_for_vehicle_at_frame.get('GapTime', float('inf'))
            
            rows_to_write.append([
                frame_id, track_id, 
                f"{wx:.2f}", f"{wy:.2f}", # Format world coordinates
                f"{px1:.1f}", f"{py1:.1f}", f"{px2:.1f}", f"{py2:.1f}", # Format pixel coordinates
                f"{ttc:.3f}" if ttc != float('inf') else 'inf',
                f"{pet:.3f}" if pet != float('inf') else 'inf',
                f"{gap_time:.3f}" if gap_time != float('inf') else 'inf'
            ])
            
    # Sort rows by frame_id, then by vehicle_id for consistent output
    rows_to_write.sort(key=lambda r: (int(r[0]), str(r[1]))) 
    
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header) # Write the header row
            writer.writerows(rows_to_write) # Write all data rows
        print(f"Results successfully saved to {output_csv_path}")
    except IOError as e: 
        print(f"Error writing to CSV file {output_csv_path}: {e}")
    except Exception as e: 
        print(f"An unexpected error occurred during CSV writing: {e}")

def draw_annotations_on_frame(frame, current_frame_id, frame_tracked_objects, frame_ssm_values):
    """
    Draws bounding boxes for tracked objects and relevant SSM information on a video frame.

    Args:
        frame (numpy.ndarray): The video frame (OpenCV BGR format) to draw on.
        current_frame_id (int): The ID of the current frame.
        frame_tracked_objects (list): List of objects tracked in this specific frame.
                                      Each item: [px1, py1, px2, py2, track_id].
        frame_ssm_values (dict): Dictionary of SSM data for vehicles in the current frame,
                                 keyed by track_id. Example:
                                 {track_id1: {'TTC': val, 'PET': val, ...}, ...}

    Returns:
        numpy.ndarray: The frame with annotations drawn.
    """
    annotated_frame = frame.copy() # Work on a copy to avoid modifying the original frame
    
    # Font and color settings for annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    id_color = (0, 255, 0)       # Green for track ID and bounding box
    ssm_text_color = (0, 255, 255) # Yellow for standard SSM text
    critical_ssm_text_color = (0, 0, 255) # Red for critical TTC values

    for obj_data in frame_tracked_objects:
        px1, py1, px2, py2, track_id = obj_data
        pt1 = (int(px1), int(py1)) # Top-left corner
        pt2 = (int(px2), int(py2)) # Bottom-right corner

        # Draw bounding box
        cv2.rectangle(annotated_frame, pt1, pt2, id_color, 2)

        # Display Track ID
        display_text_id = f"ID: {track_id}"
        cv2.putText(annotated_frame, display_text_id, (pt1[0], pt1[1] - 10), 
                    font, font_scale, id_color, font_thickness, cv2.LINE_AA)

        # Display relevant SSMs (primarily TTC for now, can be extended)
        ssm_for_this_vehicle = frame_ssm_values.get(str(track_id), {}) # Use str(track_id) for key consistency
        ttc = ssm_for_this_vehicle.get('TTC', float('inf'))
        
        display_text_ssm = ""
        current_text_color = ssm_text_color # Default color for non-critical SSMs

        if ttc < CRITICAL_TTC_THRESHOLD:
            display_text_ssm = f"TTC: {ttc:.2f}s"
            current_text_color = critical_ssm_text_color # Red for critical TTC
        elif ttc != float('inf'): # Display non-critical but finite TTC
            display_text_ssm = f"TTC: {ttc:.2f}s"
        
        # Example: Add PET or GapTime to display if needed
        # pet = ssm_for_this_vehicle.get('PET', float('inf'))
        # if pet != float('inf'): display_text_ssm += f" PET: {pet:.2f}s"

        if display_text_ssm: # Only draw if there's SSM text to show
            cv2.putText(annotated_frame, display_text_ssm, (pt1[0], pt2[1] + 20), # Position below bbox
                        font, font_scale, current_text_color, font_thickness, cv2.LINE_AA)
            
    # Display current frame ID on the frame
    frame_id_text = f"Frame: {current_frame_id}"
    cv2.putText(annotated_frame, frame_id_text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated_frame

def main_video_processor():
    """
    Main function to orchestrate video processing:
    - Parses command-line arguments.
    - Initializes video I/O, object detection (YOLO), and tracking (DeepSORT) components.
    - Attempts to use CUDA for acceleration if available.
    - Processes video frame by frame:
        - Detects vehicles.
        - Tracks detected vehicles.
        - Updates vehicle trajectories.
        - Calculates SSMs (TTC, PET, GapTime).
        - Optionally, draws annotations and saves to an output video.
    - Saves aggregated trajectory and SSM data to a CSV file.
    """
    global PIXELS_PER_METER # Allow global PIXELS_PER_METER to be updated by CLI
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Surrogate Safety Measures (SSM) Analyzer for drone video footage.")
    parser.add_argument('--input_video', type=str, required=True, 
                        help="Path to the input video file (e.g., videos/video_0058.MP4).")
    parser.add_argument('--output_csv', type=str, required=True, 
                        help="Path for the output CSV file where results will be saved.")
    parser.add_argument('--output_video', type=str, 
                        help="Optional: Path for the annotated output video file. "
                             "If not provided, no annotated video will be saved.")
    parser.add_argument('--pixels_per_meter', type=float, default=PIXELS_PER_METER,
                        help=f"Pixels per meter calibration value for converting pixel distances to meters. "
                             f"Default: {PIXELS_PER_METER}. This is a crucial parameter for metric accuracy.")
    # Add argument for ReID model path for DeepSORT
    parser.add_argument('--reid_model_path', type=str, default="osnet_x0_25_msmt17.pt",
                        help="Path or name of the ReID model checkpoint for DeepSORT. "
                             "Default: 'osnet_x0_25_msmt17.pt'. "
                             "Ensure this model is available and compatible.")
    # Add argument for YOLO model name/path
    parser.add_argument('--yolo_model', type=str, default="yolov8x.pt",
                        help="Name or path of the YOLO model to use (e.g., 'yolov8n.pt', 'yolov8x.pt', or path to custom model). "
                             "Default: 'yolov8x.pt'.")
    
    args = parser.parse_args()

    PIXELS_PER_METER = args.pixels_per_meter # Update based on CLI argument

    # --- Early CUDA Availability Check and User Feedback ---
    if TORCH_INSTALLED:
        if torch.cuda.is_available():
            print(f"Info: CUDA is available. PyTorch version: {torch.__version__}. "
                  f"CUDA version: {torch.version.cuda if torch.version.cuda else 'N/A (PyTorch built without CUDA)'}.")
            print(f"Info: Found {torch.cuda.device_count()} CUDA capable GPU(s). "
                  f"Using GPU: {torch.cuda.get_device_name(0)}.")
        else:
            print(f"Info: CUDA is not available according to PyTorch. PyTorch version: {torch.__version__}. "
                  "Models will run on CPU.")
    else:
        print("Info: PyTorch is not installed. CUDA status cannot be determined. "
              "Machine learning models (YOLO, DeepSORT ReID) cannot be loaded or run.")
        # Consider exiting here if ML models are essential and PyTorch is missing.
        # For this script, we allow proceeding without ML for non-ML parts if any.

    print(f"Info: Using Pixels Per Meter scale: {PIXELS_PER_METER}")
    print(f"Info: Starting processing for video: {args.input_video}")

    # --- Initialization Phase ---
    cap, video_fps, frame_width, frame_height = load_video(args.input_video)
    if not cap: 
        print(f"Critical Error: Failed to load video {args.input_video}. Exiting.")
        return
    print(f"Info: Video properties - FPS: {video_fps:.2f}, Width: {frame_width}, Height: {frame_height}")

    # Load YOLO model
    yolo_model = None
    yolo_device = 'cpu' # Default device for YOLO
    if ULTRALYTICS_INSTALLED:
        try:
            yolo_model_name = args.yolo_model # Use model name from CLI args
            print(f"Info: Attempting to load YOLO model ('{yolo_model_name}')...")
            yolo_model = YOLO(yolo_model_name) 
            
            # Attempt to move YOLO model to CUDA if PyTorch and CUDA are available
            if TORCH_INSTALLED and torch.cuda.is_available():
                try:
                    yolo_model.to('cuda') # Move model to GPU
                    yolo_device = 'cuda'  # Update device status
                    print(f"Info: YOLO model ('{yolo_model_name}') moved to CUDA successfully.")
                except Exception as e_cuda_yolo:
                    print(f"Warning: Failed to move YOLO model ('{yolo_model_name}') to CUDA: {e_cuda_yolo}. Using CPU.")
                    # yolo_device remains 'cpu'
            elif TORCH_INSTALLED: # PyTorch installed, but CUDA not available
                 print(f"Info: CUDA not available via PyTorch. YOLO model ('{yolo_model_name}') will run on CPU.")
            # If PyTorch not installed, Ultralytics might still work on CPU if it has fallbacks.
            # The ULTRALYTICS_INSTALLED check handles cases where YOLO cannot run at all.
            print(f"Info: YOLO model ('{yolo_model_name}') ready on device: {yolo_device}.")
        except Exception as e:
            print(f"Warning: Failed to load YOLO model ('{args.yolo_model}'): {e}. Object detection will be skipped.")
            yolo_model = None # Ensure model is None if any part of loading fails
    else:
        print("Info: Ultralytics (YOLO) library not installed. Object detection will be skipped.")

    # Initialize DeepSORT tracker
    deepsort_tracker = None
    if DEEPSORT_INSTALLED and yolo_model: # DeepSORT requires a detector
        try:
            print("Info: Attempting to initialize DeepSORT tracker...")
            # Pass ReID model path from CLI args to the initializer
            deepsort_tracker = initialize_deepsort_tracker(model_path=args.reid_model_path) 
            if deepsort_tracker: 
                print("Info: DeepSORT tracker initialized (CUDA usage determined internally based on PyTorch availability).")
            else: 
                print("Warning: Failed to initialize DeepSORT tracker. Object tracking will be skipped.")
        except Exception as e:
            print(f"Warning: Error initializing DeepSORT tracker: {e}. Object tracking will be skipped.")
    elif not yolo_model and DEEPSORT_INSTALLED:
        print("Info: YOLO model not loaded/available, so DeepSORT tracking will be skipped.")
    else: # DEEPSORT_INSTALLED is False or other dependencies missing
        print("Info: DeepSORT library or its dependencies not available. Object tracking will be skipped.")

    # Setup VideoWriter for output
    output_video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        try:
            output_video_writer = cv2.VideoWriter(args.output_video, fourcc, video_fps, (frame_width, frame_height))
            if output_video_writer.isOpened(): 
                print(f"Info: Annotated video will be saved to {args.output_video}")
            else: 
                print(f"Error: Could not open video writer for path: {args.output_video}. No annotated video will be saved.")
                output_video_writer = None 
        except Exception as e: 
            print(f"Error initializing VideoWriter for {args.output_video}: {e}. No annotated video will be saved.")
            output_video_writer = None
    
    master_vehicle_trajectories = {} # Stores all trajectory data
    master_ssm_results = {}          # Stores all SSM results
    frame_id_counter = 0             # Frame counter

    # --- Main Video Processing Loop ---
    print("\nInfo: Starting video frame processing loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Info: End of video or error reading frame at frame_id {frame_id_counter}.")
            break

        # 1. Object Detection
        current_detections = []
        if yolo_model and frame is not None:
            try: 
                current_detections = detect_vehicles_yolo(frame, yolo_model)
            except Exception as e: 
                print(f"Error during YOLO detection on frame {frame_id_counter}: {e}")
        
        # 2. Object Tracking
        current_tracked_objects = []
        if deepsort_tracker and frame is not None:
            try: 
                current_tracked_objects = track_vehicles_deepsort(frame, current_detections, deepsort_tracker)
            except Exception as e: 
                print(f"Error during DeepSORT tracking on frame {frame_id_counter}: {e}")
        
        # 3. Update Trajectories
        if current_tracked_objects:
            update_vehicle_trajectories( current_tracked_objects, frame_id_counter, PIXELS_PER_METER, master_vehicle_trajectories )
        
        # 4. Calculate SSMs
        ssm_for_current_frame = calculate_all_ssms( master_vehicle_trajectories, frame_id_counter, video_fps )
        master_ssm_results.update(ssm_for_current_frame) # Merge current frame's SSMs

        # 5. Draw Annotations and Write Video Frame
        if output_video_writer and frame is not None:
            # Prepare SSM data for drawing (keyed by track_id for this frame only)
            ssm_to_draw_on_frame = {
                track_id: data for (fid, track_id), data in ssm_for_current_frame.items() 
                if fid == frame_id_counter
            }
            annot_frame = draw_annotations_on_frame(frame.copy(), frame_id_counter, current_tracked_objects, ssm_to_draw_on_frame)
            output_video_writer.write(annot_frame)
        
        # Print progress (e.g., every 10 seconds of video)
        if frame_id_counter > 0 and video_fps > 0 and frame_id_counter % int(video_fps * 10) == 0: 
            print(f"Info: Processed frame {frame_id_counter}...")
        
        frame_id_counter += 1
        
        # # Uncomment for quick testing: process only a limited number of frames
        # if frame_id_counter >= 90: # Example: process ~3 seconds of video at 30fps
        #    print("Info: Test run, stopping early after 90 frames for brevity.")
        #    break 

    # --- Finalization Phase ---
    print(f"\nInfo: Finished processing {frame_id_counter} frames.")
    
    if cap: cap.release()
    if output_video_writer: output_video_writer.release()
    
    print(f"Info: Saving aggregated results to {args.output_csv}...")
    save_results_to_csv(master_vehicle_trajectories, master_ssm_results, args.output_csv)
    
    if args.output_video:
        # Check if video writer was successfully initialized and if file exists and has size
        if output_video_writer is not None and os.path.exists(args.output_video) and os.path.getsize(args.output_video) > 0:
             print(f"Info: Annotated video saved to {args.output_video}")
        else:
             print(f"Warning: Output video '{args.output_video}' might not have been written correctly or failed to initialize.")
    
    print("Info: Processing complete.")

if __name__ == "__main__":
    # This function now handles all operations including argument parsing.
    main_video_processor()

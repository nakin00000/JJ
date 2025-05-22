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
# calibration model is implemented.
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

# Note: VIDEO_FPS is determined from the input video itself.

import cv2
import os
import numpy as np 
import math # For float('inf'), sqrt
import csv 
import argparse # For command-line arguments

# --- Conditional Imports for Core ML/DL Libraries ---
# These flags indicate whether the necessary libraries for object detection (YOLO)
# and tracking (DeepSORT) were successfully imported.
TORCH_INSTALLED = False
ULTRALYTICS_INSTALLED = False
DEEPSORT_INSTALLED = False

try:
    import torch # PyTorch is a dependency for YOLO and DeepSORT
    TORCH_INSTALLED = True
    from ultralytics import YOLO # YOLO model for object detection
    ULTRALYTICS_INSTALLED = True
except ImportError:
    print("Info: PyTorch (torch) or Ultralytics (YOLO) not found. "
          "YOLO-based object detection will be skipped if models are attempted to load.")

try:
    # DeepSORT for object tracking
    from deep_sort_pytorch.utils.parser import get_config
    from deep_sort_pytorch.deep_sort import DeepSort
    if TORCH_INSTALLED: # DeepSORT typically relies on PyTorch
        DEEPSORT_INSTALLED = True
    else:
        print("Info: DeepSORT cannot be enabled because PyTorch (torch) is not installed.")
except ImportError:
    print("Info: deep_sort_pytorch library not found. "
          "DeepSORT-based object tracking will be skipped if models are attempted to load.")


def load_video(video_path):
    """
    Loads a video from the specified path and extracts its properties.

    Args:
        video_path (str): The file path to the video.

    Returns:
        tuple: A tuple containing:
            - cap (cv2.VideoCapture or None): The OpenCV video capture object if successful, None otherwise.
            - fps (float): Frames per second of the video. Returns a default (e.g., 30.0) if FPS is unreadable.
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
    Detects vehicles in a given frame using a YOLOv8 model.

    Args:
        frame (numpy.ndarray): The input video frame (OpenCV BGR format).
        model (ultralytics.YOLO): The pre-loaded YOLOv8 model object.

    Returns:
        list: A list of detections. Each detection is a list of the format
              [x1, y1, x2, y2, confidence, class_id], where (x1, y1) and (x2, y2)
              are the top-left and bottom-right coordinates of the bounding box.
              Returns an empty list if YOLO is not available, model is None, or no vehicles are detected.
    """
    if not ULTRALYTICS_INSTALLED or not model: 
        return [] # Skip if YOLO components are not available
        
    detections = []
    # Perform inference with verbose=False to suppress excessive YOLO output
    results = model(frame, verbose=False) 
    
    # COCO class IDs for common vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    vehicle_class_ids = [2, 3, 5, 7] 

    if results and results[0].boxes:
        for box in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            if int(class_id) in vehicle_class_ids:
                detections.append([x1, y1, x2, y2, confidence, int(class_id)])
    return detections

def initialize_deepsort_tracker(model_path="osnet_x0_25_msmt17.pt"):
    """
    Initializes the DeepSORT tracker with a specified ReID model.

    Args:
        model_path (str): Path or name of the ReID model checkpoint file
                          (e.g., "osnet_x0_25_msmt17.pt"). The DeepSORT library
                          might search for this in predefined paths or require a full path
                          depending on its specific implementation.

    Returns:
        DeepSort object or None: The initialized DeepSORT tracker object if successful,
                                 None otherwise.
    """
    if not DEEPSORT_INSTALLED: 
        print("Info: DeepSORT not installed or not enabled; skipping tracker initialization.")
        return None
    try:
        cfg = get_config() # Load DeepSORT configurations
        # The deep_sort_pytorch library usually handles default config loading.
        # If a specific config file is needed (e.g., "deep_sort_pytorch/configs/deep_sort.yaml"),
        # ensure it's accessible or provide the correct path.
        
        use_cuda_flag = torch.cuda.is_available() if TORCH_INSTALLED else False

        tracker = DeepSort( 
            model_path, 
            max_dist=cfg.DEEPSORT.MAX_DIST, 
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, 
            n_init=cfg.DEEPSORT.N_INIT, 
            nn_budget=cfg.DEEPSORT.NN_BUDGET, 
            use_cuda=use_cuda_flag 
        )
        return tracker
    except Exception as e:
        print(f"Error initializing DeepSORT tracker: {e}")
        print("Ensure ReID model weights (e.g., osnet_x0_25_msmt17.pt) are available "
              "and the path is correct or resolvable by the DeepSORT library.")
        return None

def track_vehicles_deepsort(frame, detections, tracker):
    """
    Tracks detected vehicles using the DeepSORT algorithm.

    Args:
        frame (numpy.ndarray): The current video frame (OpenCV BGR format).
        detections (list): A list of detections from YOLO, where each detection is
                           [x1, y1, x2, y2, confidence, class_id].
        tracker (DeepSort object): The initialized DeepSORT tracker.

    Returns:
        list: A list of tracked objects. Each object is a list of the format
              [x1, y1, x2, y2, track_id], where (x1,y1) and (x2,y2) are
              pixel coordinates of the bounding box for the tracked object,
              and track_id is its unique identifier (as a string).
              Returns an empty list if DeepSORT is not available, tracker is None,
              or no objects are successfully tracked.
    """
    if not DEEPSORT_INSTALLED or not tracker: 
        return [] 
    
    if not detections: # If no detections, update tracker with empty info and return
        tracker.update(np.array([]), np.array([]), np.array([]), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return []

    xywhs = []; confs = []; oids = []
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        # DeepSORT expects bounding boxes in [center_x, center_y, width, height] format
        xywhs.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confs.append(conf)
        oids.append(class_id) # Original class ID from detector
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # DeepSORT often uses RGB for ReID
    
    np_xywhs = np.array(xywhs, dtype=float)
    np_confs = np.array(confs, dtype=float)
    np_oids = np.array(oids, dtype=int) 

    tracker.update(np_xywhs, np_confs, np_oids, rgb_frame)
    
    tracked_output = []
    # tracker.outputs usually contains [x1, y1, x2, y2, track_id, class_id (optional), conf (optional)]
    for track_data in tracker.outputs:
        if len(track_data) >= 5: # Ensure at least bbox and track_id are present
            x1, y1, x2, y2, track_id = track_data[0:5]
            # Ensure coordinates are integers for consistency and drawing
            tracked_output.append([int(x1), int(y1), int(x2), int(y2), str(track_id)])
    return tracked_output

def calibrate_pixel_to_meters(pixel_x, pixel_y, current_pixels_per_meter):
    """
    Converts pixel coordinates to world coordinates (in meters) using a simple scale factor.

    Args:
        pixel_x (float or int): The x-coordinate in pixels.
        pixel_y (float or int): The y-coordinate in pixels.
        current_pixels_per_meter (float): The scale factor (pixels per meter).
                                         Must be accurately calibrated.

    Returns:
        tuple: (world_x, world_y) representing coordinates in meters.
               The origin (0,0) in pixel space maps to (0,0) in world space here.
               Y-axis direction might need inversion depending on world coordinate setup.

    Raises:
        ValueError: If current_pixels_per_meter is not positive.
    """
    if current_pixels_per_meter <= 0: 
        raise ValueError("pixels_per_meter_scale must be positive.")
    world_x = pixel_x / current_pixels_per_meter
    world_y = pixel_y / current_pixels_per_meter # Simple scaling; might need y-inversion for some world coord systems
    return world_x, world_y

def update_vehicle_trajectories(current_tracked_objects, frame_id, current_pixels_per_meter, current_trajectories_dict):
    """
    Updates a given dictionary of vehicle trajectories with new tracking data for the current frame.
    Each trajectory point stores frame ID, world coordinates (calibrated from bottom-center of pixel bbox),
    and the full pixel bounding box.

    Args:
        current_tracked_objects (list): List of objects tracked in the current frame.
                                        Each item: [x1_pix, y1_pix, x2_pix, y2_pix, track_id].
        frame_id (int): The ID of the current frame.
        current_pixels_per_meter (float): The current pixel-to-meter scale factor.
        current_trajectories_dict (dict): The dictionary to update.
            Format: {track_id: [(frame_id, world_x, world_y, px1, py1, px2, py2), ...], ...}
            This dictionary is modified in-place.
    """
    for obj_data in current_tracked_objects:
        x1_pix, y1_pix, x2_pix, y2_pix, track_id_str = obj_data
        
        # Calculate bottom-center of the pixel bounding box for world coordinate mapping
        pixel_center_x_for_world = (x1_pix + x2_pix) / 2.0
        pixel_center_y_for_world = float(y2_pix) # Using bottom of the bounding box
        
        try:
            world_x, world_y = calibrate_pixel_to_meters(
                pixel_center_x_for_world, 
                pixel_center_y_for_world, 
                current_pixels_per_meter
            )
            
            track_id_key = str(track_id_str) # Ensure track_id is a string for dictionary key consistency

            if track_id_key not in current_trajectories_dict:
                current_trajectories_dict[track_id_key] = []
            
            # Store frame_id, world coords, and the original pixel bounding box
            current_trajectories_dict[track_id_key].append(
                (frame_id, world_x, world_y, 
                 float(x1_pix), float(y1_pix), float(x2_pix), float(y2_pix)) 
            )
        except ValueError as e:
            print(f"Warning: Calibration error for track_id {track_id_str}, frame {frame_id}: {e}")
        except Exception as e: # Catch any other unexpected errors during update
            print(f"Warning: Trajectory update error for track_id {track_id_str}, frame {frame_id}: {e}")

def _get_trajectory_point(trajectory_points, frame_id_to_find):
    """
    Helper function to find a specific data point in a vehicle's trajectory list by its frame_id.
    Searches backwards, assuming recent points are more likely to be needed.

    Args:
        trajectory_points (list): A list of tuples, where each tuple is a trajectory point:
                                  (frame_id, world_x, world_y, px1, py1, px2, py2).
        frame_id_to_find (int): The frame ID of the point to retrieve.

    Returns:
        tuple or None: The trajectory point tuple if found, otherwise None.
    """
    for point in reversed(trajectory_points):
        if point[0] == frame_id_to_find:
            return point
    return None

def _get_vehicle_velocity(trajectory_full, current_frame_id, fps, 
                           velocity_estimation_window_frames=VELOCITY_ESTIMATION_WINDOW_FRAMES):
    """
    Estimates a vehicle's velocity (vx, vy) in meters per second using its trajectory.
    Velocity is calculated based on the change in world coordinates between the current
    point and the most recent previous point within the estimation window.

    Args:
        trajectory_full (list): The vehicle's complete list of trajectory points.
        current_frame_id (int): The frame ID for which to estimate velocity.
        fps (float): Frames per second of the video.
        velocity_estimation_window_frames (int): Max number of past frames to search for a
                                                 previous point for velocity calculation.

    Returns:
        tuple (vx, vy) or (None, None): Estimated velocity components (vx, vy) in m/s,
                                        or (None, None) if velocity cannot be determined
                                        (e.g., insufficient trajectory history).
    """
    cp = _get_trajectory_point(trajectory_full, current_frame_id) # Current point
    if not cp: 
        return None, None # No data for current frame

    pp = None # Previous point
    for i in range(1, velocity_estimation_window_frames + 1):
        prev_frame_id = current_frame_id - i
        if prev_frame_id < 0: # Reached before the start of the video
            break
        pp_candidate = _get_trajectory_point(trajectory_full, prev_frame_id)
        if pp_candidate:
            pp = pp_candidate
            break # Found the most recent previous point in the window
            
    if not pp: 
        return None, None # No suitable previous point found within the window

    # Extract world coordinates (indices 1 and 2) from points
    x_curr, y_curr = cp[1], cp[2]
    x_prev, y_prev = pp[1], pp[2]
    
    frame_diff = cp[0] - pp[0] # Difference in frame numbers
    if frame_diff <= 0: 
        # Should not happen if cp and pp are distinct and correctly ordered
        return None, None 
    
    time_diff_seconds = frame_diff / fps
    vx = (x_curr - x_prev) / time_diff_seconds
    vy = (y_curr - y_prev) / time_diff_seconds
    return vx, vy

def check_bounding_box_overlap(box1, box2):
    """
    Checks if two bounding boxes overlap.

    Args:
        box1 (tuple): Coordinates (x1, y1, x2, y2) of the first box.
        box2 (tuple): Coordinates (x1, y1, x2, y2) of the second box.
                      Assumes x2 > x1 and y2 > y1 for both boxes.

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    # Unpack for clarity (optional, direct indexing box[0] etc. is also fine)
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    # Check for non-overlap conditions (if one box is to the left, right, above, or below the other)
    if box1_x2 < box2_x1 or box1_x1 > box2_x2 or \
       box1_y2 < box2_y1 or box1_y1 > box2_y2:
        return False
    return True # If none of the non-overlap conditions are met, boxes overlap

def calculate_ttc_for_pair(traj1_full, traj2_full, current_frame_id, fps):
    """
    Calculates Time-to-Collision (TTC) in seconds between two vehicles.
    Assumes a constant velocity model for prediction.

    Args:
        traj1_full (list): Full trajectory data for vehicle 1.
        traj2_full (list): Full trajectory data for vehicle 2.
        current_frame_id (int): The current frame ID for calculation.
        fps (float): Video frames per second.

    Returns:
        float: Calculated TTC in seconds. Returns float('inf') if no collision is
               predicted (e.g., vehicles moving apart, parallel, or data is insufficient).
    """
    # Get current positions for both vehicles
    cp1 = _get_trajectory_point(traj1_full, current_frame_id)
    cp2 = _get_trajectory_point(traj2_full, current_frame_id)
    if not cp1 or not cp2: 
        return float('inf') # One or both vehicles not present at current_frame_id

    # Get velocities for both vehicles
    vx1, vy1 = _get_vehicle_velocity(traj1_full, current_frame_id, fps)
    vx2, vy2 = _get_vehicle_velocity(traj2_full, current_frame_id, fps)

    if vx1 is None or vx2 is None: 
        return float('inf') # Cannot determine velocity for one or both

    # Current world positions (indices 1 and 2 of trajectory points)
    x1_curr, y1_curr = cp1[1], cp1[2]
    x2_curr, y2_curr = cp2[1], cp2[2]
    
    # Relative positions and velocities
    dx = x1_curr - x2_curr  # Difference in x positions
    dy = y1_curr - y2_curr  # Difference in y positions
    dvx = vx1 - vx2         # Difference in x velocities
    dvy = vy1 - vy2         # Difference in y velocities
    
    # Dot product of relative position and relative velocity vector components
    # If positive, vehicles are moving apart or parallel (not on collision course along this line).
    dot_product_rel_pos_vel = dvx * dx + dvy * dy
    if dot_product_rel_pos_vel >= 0: 
        return float('inf')
    
    # Squared distance between vehicle centers
    distance_squared = dx**2 + dy**2
    if distance_squared == 0: # Vehicles are at the same point (already collided)
        return 0.0 
    
    # TTC = -(distance_squared) / (relative_velocity_vector DOT relative_position_vector)
    # Note: The denominator is negative if vehicles are closing.
    ttc = distance_squared / -dot_product_rel_pos_vel

    # Return TTC only if it's non-negative (collision in future or now)
    return ttc if ttc >= 0 else float('inf')

def calculate_all_ssms(current_vehicle_trajectories, current_frame_id, fps):
    """
    Calculates all selected Surrogate Safety Measures (SSMs) for relevant vehicle pairs.
    Currently includes: Time-to-Collision (TTC), Post-Encroachment Time (PET),
    and Gap Time (Time Headway).

    Args:
        current_vehicle_trajectories (dict): All known vehicle trajectories up to the current frame.
            Format: {track_id: [(frame_id, world_x, world_y, px1, py1, px2, py2), ...]}
        current_frame_id (int): The frame ID for which SSMs are being calculated.
        fps (float): Video frames per second.

    Returns:
        dict: SSM results for the current frame.
            Format: {(frame_id, track_id): {'TTC': value, 'PET': value, 'GapTime': value}, ...}
            Each SSM value is in seconds or float('inf') if not applicable/calculable.
    """
    ssm_results = {} # Stores SSMs for vehicles involved in interactions this frame
    
    # Identify vehicles present in the current frame
    active_track_ids = [ track_id for track_id, trajectory in current_vehicle_trajectories.items() 
                         if _get_trajectory_point(trajectory, current_frame_id) ]

    # Iterate through unique pairs of active vehicles
    for i in range(len(active_track_ids)):
        for j in range(i + 1, len(active_track_ids)):
            id1, id2 = active_track_ids[i], active_track_ids[j]
            traj1_full = current_vehicle_trajectories[id1]
            traj2_full = current_vehicle_trajectories[id2]

            # Initialize SSM dictionary for both vehicles if not already present
            key1 = (current_frame_id, id1); key2 = (current_frame_id, id2)
            for key in [key1, key2]:
                if key not in ssm_results: 
                    ssm_results[key] = {'TTC': float('inf'), 'PET': float('inf'), 'GapTime': float('inf')}

            # --- TTC Calculation ---
            current_ttc = calculate_ttc_for_pair(traj1_full, traj2_full, current_frame_id, fps)
            # Update TTC for both vehicles involved in this pair interaction
            ssm_results[key1]['TTC'] = min(ssm_results[key1]['TTC'], current_ttc)
            ssm_results[key2]['TTC'] = min(ssm_results[key2]['TTC'], current_ttc)

            # --- PET Calculation ---
            current_pet_for_pair = float('inf')
            point1_curr = _get_trajectory_point(traj1_full, current_frame_id) # Re-fetch, already got in TTC but good for clarity
            point2_curr = _get_trajectory_point(traj2_full, current_frame_id)
            
            if point1_curr and point2_curr: # Both vehicles must exist in current frame
                # Pixel BBoxes are at indices 3,4,5,6
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
                world_x1, world_y1 = point1_curr[1], point1_curr[2] # World coords
                world_x2, world_y2 = point2_curr[1], point2_curr[2]

                vx1, vy1 = _get_vehicle_velocity(traj1_full, current_frame_id, fps)
                vx2, vy2 = _get_vehicle_velocity(traj2_full, current_frame_id, fps)

                speed1 = math.sqrt(vx1**2 + vy1**2) if vx1 is not None and vy1 is not None else 0.0
                speed2 = math.sqrt(vx2**2 + vy2**2) if vx2 is not None and vy2 is not None else 0.0
                
                follower_id_for_gap = None
                distance_gap = float('inf')
                follower_speed_for_gap = 0.0

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
                    # Calculate GapTime if follower is moving (speed > 0.1 m/s threshold)
                    if follower_speed_for_gap > 0.1: 
                        gap_time_value = distance_gap / follower_speed_for_gap
                    
                    key_follower = (current_frame_id, follower_id_for_gap)
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

        # Display relevant SSMs (primarily TTC for now)
        ssm_for_this_vehicle = frame_ssm_values.get(str(track_id), {}) # Use str(track_id) for key consistency
        ttc = ssm_for_this_vehicle.get('TTC', float('inf'))
        
        display_text_ssm = ""
        current_text_color = ssm_text_color # Default color for non-critical SSMs

        if ttc < CRITICAL_TTC_THRESHOLD:
            display_text_ssm = f"TTC: {ttc:.2f}s"
            current_text_color = critical_ssm_text_color # Red for critical TTC
        elif ttc != float('inf'): # Display non-critical but finite TTC
            display_text_ssm = f"TTC: {ttc:.2f}s"
        
        # Add other SSMs to display_text_ssm if desired, e.g., PET, GapTime
        # Example:
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
    Main function to process a video: parse arguments, load models, process frames,
    calculate SSMs, and save results.
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
                             f"Default: {PIXELS_PER_METER}. This is a crucial parameter.")
    
    args = parser.parse_args()

    # Update global PIXELS_PER_METER if provided via command line
    # This is one of the few globals modified by CLI; most params are passed around.
    PIXELS_PER_METER = args.pixels_per_meter
    print(f"Info: Using Pixels Per Meter scale: {PIXELS_PER_METER}")

    # --- Initialization Phase ---
    print(f"Info: Starting processing for video: {args.input_video}")
    cap, video_fps, frame_width, frame_height = load_video(args.input_video)
    if not cap:
        print(f"Critical Error: Failed to load video {args.input_video}. Exiting.")
        return

    print(f"Info: Video properties - FPS: {video_fps:.2f}, Width: {frame_width}, Height: {frame_height}")

    # Load YOLO model for object detection
    yolo_model = None
    if ULTRALYTICS_INSTALLED:
        try:
            print("Info: Attempting to load YOLOv8 model (yolov8n.pt)...")
            yolo_model = YOLO('yolov8n.pt') # Standard model name, should be auto-downloaded if not present
            print("Info: YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load YOLOv8 model: {e}. Object detection will be skipped.")
    else:
        print("Info: Ultralytics (YOLO) library not installed. Object detection will be skipped.")

    # Initialize DeepSORT tracker
    deepsort_tracker = None
    if DEEPSORT_INSTALLED and yolo_model: # DeepSORT is typically used with a detector
        try:
            print("Info: Attempting to initialize DeepSORT tracker...")
            deepsort_tracker = initialize_deepsort_tracker() # Uses default ReID model path
            if deepsort_tracker:
                 print("Info: DeepSORT tracker initialized successfully.")
            else:
                 print("Warning: Failed to initialize DeepSORT tracker. Object tracking will be skipped.")
        except Exception as e:
            print(f"Warning: Error initializing DeepSORT tracker: {e}. Object tracking will be skipped.")
    elif not yolo_model and DEEPSORT_INSTALLED:
        print("Info: YOLO model not loaded, so DeepSORT tracking will be skipped.")
    else: # DEEPSORT_INSTALLED is False
        print("Info: DeepSORT library not installed or not enabled. Object tracking will be skipped.")

    # Setup VideoWriter for output annotated video if path is provided
    output_video_writer = None
    if args.output_video:
        # Define video codec (e.g., mp4v for .mp4, XVID for .avi)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        try:
            output_video_writer = cv2.VideoWriter(args.output_video, fourcc, video_fps, (frame_width, frame_height))
            if not output_video_writer.isOpened():
                print(f"Error: Could not open video writer for path: {args.output_video}")
                output_video_writer = None # Ensure it's None if opening failed
            else:
                print(f"Info: Annotated video will be saved to {args.output_video}")
        except Exception as e:
            print(f"Error initializing VideoWriter for {args.output_video}: {e}")
            output_video_writer = None
    
    # Data storage for the entire video
    master_vehicle_trajectories = {} # Stores all trajectory data: {track_id: [points...]}
    master_ssm_results = {}          # Stores all SSM results: {(frame_id, track_id): {ssms...}}
    
    frame_id_counter = 0 # Initialize frame counter

    # --- Main Video Processing Loop ---
    print("\nInfo: Starting video frame processing loop...")
    while cap.isOpened():
        ret, frame = cap.read() # Read a frame from the video
        if not ret:
            print(f"Info: End of video or error reading frame at frame_id {frame_id_counter}.")
            break

        # 1. Object Detection (YOLO)
        current_detections = []
        if yolo_model and frame is not None:
            try:
                current_detections = detect_vehicles_yolo(frame, yolo_model)
            except Exception as e: # Catch potential errors during detection
                print(f"Error during YOLO detection on frame {frame_id_counter}: {e}")
        
        # 2. Object Tracking (DeepSORT)
        current_tracked_objects = [] # List of [x1,y1,x2,y2, track_id] for current frame
        if deepsort_tracker and frame is not None:
            try:
                # Pass current_detections (even if empty) to the tracker
                current_tracked_objects = track_vehicles_deepsort(frame, current_detections, deepsort_tracker)
            except Exception as e: # Catch potential errors during tracking
                print(f"Error during DeepSORT tracking on frame {frame_id_counter}: {e}")
        
        # 3. Update Trajectories
        if current_tracked_objects: # Only update if there are objects tracked in this frame
            update_vehicle_trajectories(
                current_tracked_objects, 
                frame_id_counter, 
                PIXELS_PER_METER, # Use current (possibly CLI-overridden) value
                master_vehicle_trajectories
            )

        # 4. Calculate SSMs for the current frame
        # This uses the cumulative master_vehicle_trajectories up to the current frame
        ssm_for_current_frame_pairs = calculate_all_ssms(
            master_vehicle_trajectories, 
            frame_id_counter, 
            video_fps # Use actual video FPS
        )
        # Merge results for the current frame into the master SSM log
        master_ssm_results.update(ssm_for_current_frame_pairs)

        # 5. Draw Annotations and Write to Output Video (if applicable)
        if output_video_writer and frame is not None:
            # Prepare SSM data specifically for drawing (keyed by track_id for this frame)
            ssm_to_draw_on_frame = {
                track_id: data for (fid, track_id), data in ssm_for_current_frame_pairs.items() 
                if fid == frame_id_counter
            }
            annot_frame = draw_annotations_on_frame(
                frame.copy(), # Draw on a copy of the frame
                frame_id_counter, 
                current_tracked_objects, 
                ssm_to_draw_on_frame
            )
            output_video_writer.write(annot_frame)
        
        # Print progress periodically
        if frame_id_counter > 0 and frame_id_counter % int(video_fps * 10) == 0 : # Every 10 seconds of video
             print(f"Info: Processed frame {frame_id_counter}...")
        
        frame_id_counter += 1
        
        # # Uncomment for quick testing: process only a few frames
        # if frame_id_counter >= 90 : # Example: process ~3 seconds of video at 30fps
        #    print("Info: Stopping early for a quick test run after 90 frames.")
        #    break

    # --- Finalization Phase ---
    print(f"\nInfo: Finished processing {frame_id_counter} frames.")
    
    # Release video capture and writer resources
    if cap: 
        cap.release()
    if output_video_writer: 
        output_video_writer.release()
    # cv2.destroyAllWindows() # Only if cv2.imshow was used
    
    # Save all collected trajectory and SSM data to CSV
    print(f"Info: Saving results to {args.output_csv}...")
    save_results_to_csv(master_vehicle_trajectories, master_ssm_results, args.output_csv)
    
    if args.output_video:
        if os.path.exists(args.output_video) and os.path.getsize(args.output_video) > 0:
             print(f"Info: Annotated video saved to {args.output_video}")
        else:
             print(f"Warning: Output video was requested ({args.output_video}) but might not have been written correctly.")
    
    print("Info: Processing complete.")


if __name__ == "__main__":
    # Call the main video processing function when script is executed
    main_video_processor()

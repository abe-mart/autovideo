import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import tempfile
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import json
import os


# --- CHANGE 1: Set page layout to wide mode ---
# This should be the first Streamlit command in your script.
st.set_page_config(layout="wide")

# Set the title of the Streamlit app
st.title("Video Picture Frame Inserter")

st.write("""
Upload a video with a picture frame. On the first frame, draw a rectangle 
around the **inside** of the empty frame. Then, upload an image to place inside it.
""")

@st.fragment
def corner_selector_fragment(frame_to_display, key_suffix):
    """
    A self-contained fragment to handle the corner selection for a single frame.
    """
    display_img, scale_factor = prepare_display_image(frame_to_display)
        
    image_to_draw_on = np.array(display_img)
    # Draw points from our temporary list
    for i, corner in enumerate(st.session_state.temp_corners):
        display_x = int(corner[0] * scale_factor)
        display_y = int(corner[1] * scale_factor)
        cv2.circle(image_to_draw_on, (display_x, display_y), 5, (34, 139, 34), -1)
        cv2.putText(image_to_draw_on, str(i + 1), (display_x + 8, display_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Use a unique key for the component for each shot to avoid state conflicts
    value = streamlit_image_coordinates(image_to_draw_on, key=f"shot_selector_{key_suffix}")

    if value:
        original_x = value['x'] / scale_factor
        original_y = value['y'] / scale_factor
        new_corner = (original_x, original_y)
        if new_corner not in st.session_state.temp_corners and len(st.session_state.temp_corners) < 4:
            st.session_state.temp_corners.append(new_corner)
            # Rerun the fragment to show the new point instantly
            st.rerun()

def detect_shot_boundaries_pysd(video_path):
    """
    Analyzes a video to detect shot boundaries using the modern PySceneDetect v0.6+ API.
    """
    try:
        # Use open_video as a context manager for automatic resource handling
        video = open_video(video_path)
        scene_manager = SceneManager()
        # ContentDetector is still the right tool for finding hard cuts.
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        
        # The detect_scenes call is simpler now, just pass the video object.
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            return [0]

        # The rest of the logic to get the start frames remains the same
        shot_boundaries = [scene[0].get_frames() for scene in scene_list]
        if 0 not in shot_boundaries:
            shot_boundaries.insert(0, 0)
            
        return sorted(list(set(shot_boundaries)))

    except Exception as e:
        st.error(f"PySceneDetect failed: {e}")
        # It's good practice to print the exception for debugging
        print(e)
        return None

def get_frame(video_path, frame_number):
    """
    Grabs a specific frame from a video file.
    """
    vid_capture = cv2.VideoCapture(video_path)
    if not vid_capture.isOpened():
        return None
    # Set the video's position to the desired frame
    vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = vid_capture.read()
    vid_capture.release()
    if success:
        return frame
    return None

# We also need a helper function to prepare the image for display
def prepare_display_image(frame, max_width=1200):
    original_height, original_width, _ = frame.shape
    if original_width > max_width:
        scale_factor = max_width / original_width
    else:
        scale_factor = 1.0
    
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    display_img = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
    
    return display_img, scale_factor

# 1. Video Upload
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    # We need to save the uploaded file to a temporary location so OpenCV can read it.
    st.session_state.video_filename = uploaded_video.name
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.success(f"Video '{uploaded_video.name}' uploaded successfully.")

    with st.spinner("Analyzing video for shot changes..."):
        # --- NEW: Call the shot detection function ---
        shot_starts = detect_shot_boundaries_pysd(video_path)
        st.session_state.shot_starts = shot_starts

    if shot_starts is not None:
        st.session_state.shot_starts = shot_starts
        st.success(f"Video analysis complete! Found {len(st.session_state.shot_starts)} distinct shot(s).")
        st.info("Next, we will select the corners for the first frame of each shot.")
        st.write("Shots begin at frames:", st.session_state.shot_starts)
    else:
        st.error("Could not process the video. Please try another file.")

    # Initialize state for the selection process
    if 'shot_corners' not in st.session_state:
        st.session_state.shot_corners = {}
    if 'current_shot_index' not in st.session_state:
        st.session_state.current_shot_index = 0
    if 'temp_corners' not in st.session_state:
        st.session_state.temp_corners = []

    shot_starts = st.session_state.shot_starts
    num_shots = len(shot_starts)
    current_idx = st.session_state.current_shot_index

    # Check if there are still shots that need corners defined
    if current_idx < num_shots:
        
        current_frame_num = shot_starts[current_idx]

        st.subheader(f"Step 2: Define Corners for Shot {current_idx + 1}/{num_shots}")
        st.write(f"This shot starts at frame **{current_frame_num}**.")

        # Get the specific frame for the current shot
        frame_for_selection = get_frame(video_path, current_frame_num)

        if frame_for_selection is not None:
            # Call the fragment to handle the interactive selection
            corner_selector_fragment(frame_for_selection, current_idx)

            # --- UI for confirmation and reset (remains outside the fragment) ---
            col1, col2 = st.columns([1,3])
            with col1:
                if st.button("Reset corners for this shot"):
                    st.session_state.temp_corners = []
                    st.rerun()

            if len(st.session_state.temp_corners) == 4:
                with col2:
                    # Use a prominent button to confirm and advance
                    if st.button("✅ Confirm & Go to Next Shot", type="primary"):
                        st.session_state.shot_corners[current_frame_num] = st.session_state.temp_corners
                        st.session_state.current_shot_index += 1
                        st.session_state.temp_corners = []
                        st.rerun() # Trigger a full rerun to load the next shot UI

    else:
        st.success("All corners defined for all shots!")
        st.info("You can now process the video to track these corners and optionally generate a debug video.")

        # --- UI for Final Processing ---

        # Add a checkbox to control whether the debug video is created
        # create_debug_video = st.checkbox("Create debug video with corner overlays")
        create_debug_video = True

        # Use a button to trigger the final tracking process
        if st.button("👁️ Track Corners Through All Shots", type="primary"):
            
            # --- TRACKING & DATA COLLECTION / DEBUG VIDEO RENDERING ---
            with st.spinner("Processing video... This may take a while."):
                # 1. SETUP
                vid_capture = cv2.VideoCapture(video_path)
                total_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # This dictionary will store our final results
                final_coordinates = {}

                # Setup for debug video if requested
                debug_video_writer = None
                if create_debug_video:
                    original_w = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    original_h = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = vid_capture.get(cv2.CAP_PROP_FPS)

                    # --- NEW: Define and calculate scaled-down dimensions ---
                    DEBUG_VIDEO_WIDTH = 800
                    scale_factor = DEBUG_VIDEO_WIDTH / original_w
                    debug_w = DEBUG_VIDEO_WIDTH
                    debug_h = int(original_h * scale_factor)
                    
                    fps = vid_capture.get(cv2.CAP_PROP_FPS)
                    debug_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                    # --- CHANGE 1: Use a more browser-compatible codec ---
                    # 'avc1' corresponds to H.264, the web standard.
                    fourcc = cv2.VideoWriter_fourcc(*'VP80')
                    
                    debug_video_writer = cv2.VideoWriter(debug_output_path, fourcc, fps, (debug_w, debug_h))
                    
                    # --- CHANGE 2: Add error checking ---
                    if not debug_video_writer.isOpened():
                        st.error("Error: Could not open video writer. Your OpenCV build may not support the 'avc1' (H.264) codec. Try another option.")
                        # We'll stop this execution but keep the app running.
                        st.stop()
                    
                    st.session_state.debug_video_path = debug_output_path
                    debug_video_writer = cv2.VideoWriter(debug_output_path, fourcc, fps, (debug_w, debug_h))
                    st.session_state.debug_video_path = debug_output_path


                # Tracking parameters and state variables
                lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                prev_gray = None
                prev_pts = None
                tracking_active = False
                
                # 2. MAIN TRACKING LOOP
                for frame_num in range(total_frames):
                    success, frame = vid_capture.read()
                    if not success:
                        break

                    # --- SHOT AWARENESS & TRACKING LOGIC ---
                    if frame_num in st.session_state.shot_corners:
                        tracking_active = True
                        initial_corners = np.array(st.session_state.shot_corners[frame_num], dtype=np.float32)
                        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prev_pts = initial_corners.reshape(-1, 1, 2)
                        current_pts = prev_pts
                    elif tracking_active:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
                        
                        good_pts_count = np.sum(status) if status is not None else 0
                        if next_pts is not None and good_pts_count >= 3:
                            current_pts = next_pts
                            prev_gray = frame_gray.copy()
                            prev_pts = current_pts.reshape(-1, 1, 2)
                        else:
                            tracking_active = False
                    
                    # --- DATA COLLECTION & DEBUG VIDEO DRAWING ---
                    if tracking_active:
                        final_coordinates[frame_num] = current_pts.reshape(4, 2).tolist()
                        
                        # If debug mode is on, draw the points on the frame
                        if create_debug_video:
                            for i, point in enumerate(current_pts):
                                x, y = int(point[0][0]), int(point[0][1])
                                # Draw a colored circle for each of the 4 points
                                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][i]
                                cv2.circle(frame, (x, y), 7, color, -1)

                    # --- NEW: Resize frame before writing ---
                    if create_debug_video and debug_video_writer is not None:
                        # Resize the frame (which may have points drawn on it) to the target debug size
                        resized_frame = cv2.resize(frame, (debug_w, debug_h), interpolation=cv2.INTER_AREA)
                        debug_video_writer.write(resized_frame)
                
                # 3. CLEANUP
                vid_capture.release()
                if create_debug_video and debug_video_writer is not None:
                    debug_video_writer.release()
                
                st.session_state.final_coordinates = final_coordinates
                
            st.success(f"Tracking complete! Corner data collected for {len(final_coordinates)} frames.")


        # --- UI AFTER PROCESSING ---

        # Display the download button for the coordinate data
        if 'final_coordinates' in st.session_state:
            

            # --- NEW: Generate filename based on the original video's name ---
            # Get the original filename without its extension
            base_name, _ = os.path.splitext(st.session_state.video_filename)
            # Create the new filename for the JSON file
            json_filename = f"{base_name}_coordinates.json"

            json_data = json.dumps(st.session_state.final_coordinates, indent=4)
            st.download_button(
                label="⬇️ Download Coordinates as JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
            )

        # Display the debug video if it was created
        if 'debug_video_path' in st.session_state and create_debug_video:
            col = st.columns([0.6,0.4])
            with col[0]:
                st.subheader("Debug Video with Corner Overlays")
                st.video(st.session_state.debug_video_path)

                    
else:
    st.info("Please upload a video file to begin.")



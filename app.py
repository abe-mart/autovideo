import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import tempfile
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector


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
def corner_selection_fragment(display_img_pil, scale_factor):
    """
    This fragment handles the image display, drawing, and click handling.
    It reruns on its own without reloading the whole page.
    """
    # Convert the display PIL image to an OpenCV-compatible format for drawing
    image_to_draw_on = np.array(display_img_pil)

    # Draw existing corners on the image
    for i, corner in enumerate(st.session_state.corners):
        display_x = int(corner[0] * scale_factor)
        display_y = int(corner[1] * scale_factor)
        cv2.circle(image_to_draw_on, (display_x, display_y), radius=5, color=(34, 139, 34), thickness=-1)
        cv2.putText(image_to_draw_on, str(i + 1), (display_x + 8, display_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Use the component with the image that has points drawn on it
    value = streamlit_image_coordinates(image_to_draw_on, key="pil_fragment")

    # When a click occurs, 'value' is populated.
    if value:
        clicked_x, clicked_y = value['x'], value['y']
        original_x = clicked_x / scale_factor
        original_y = clicked_y / scale_factor
        new_corner = (original_x, original_y)
        
        # Add the corner if it's new and we need more.
        if new_corner not in st.session_state.corners:
            if len(st.session_state.corners) < 4:
                st.session_state.corners.append(new_corner)
                st.rerun()  # Rerun to update the display with the new corner

def detect_shot_boundaries_pysd(video_path):
    """
    Analyzes a video to detect shot boundaries using the PySceneDetect library.

    Args:
        video_path (str): The path to the video file.

    Returns:
        list: A list of frame numbers where new shots begin. Always includes frame 0.
    """
    try:
        # Create a video_manager to open the video
        video_manager = VideoManager([video_path])
        
        # Create a scene_manager to manage the scenes
        scene_manager = SceneManager()

        # Add the ContentDetector algorithm to the scene_manager.
        # The threshold is the change in content required to trigger a cut.
        # A value of 27-30 is a good starting point.
        scene_manager.add_detector(ContentDetector(threshold=27.0))

        # Perform scene detection
        video_manager.set_downscale_factor(2) # Downscale for faster processing
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        # The result is a list of scenes (start_time, end_time).
        # We just need the start frame of each scene.
        if not scene_list:
            return [0] # No cuts detected, so the whole video is one shot

        shot_boundaries = [scene[0].get_frames() for scene in scene_list]
        
        # Ensure frame 0 is always included, even if the first detected scene is later
        if 0 not in shot_boundaries:
            shot_boundaries.insert(0, 0)
            
        return sorted(list(set(shot_boundaries))) # Return a sorted list of unique frame numbers

    except Exception as e:
        st.error(f"PySceneDetect failed: {e}")
        return None
    finally:
        # Always ensure the video manager is released
        if video_manager is not None:
            video_manager.release()

# 1. Video Upload
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    # We need to save the uploaded file to a temporary location so OpenCV can read it.
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
    
    # # Open the video file
    # vid_capture = cv2.VideoCapture(video_path)
    # success, first_frame = vid_capture.read()

    # if not success:
    #     st.error("Could not read the first frame of the video.")
    # else:
    #     st.subheader("Step 1: Click the 4 corners of the frame")
    #     st.write("Click on the image to select the corners in **clockwise order**: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")

    #     # --- CHANGE 2: Image Scaling ---
    #     original_height, original_width, _ = first_frame.shape
        
    #     # Define a max width for the display image
    #     MAX_DISPLAY_WIDTH = 1200

    #     # Calculate scaling factor
    #     if original_width > MAX_DISPLAY_WIDTH:
    #         scale_factor = MAX_DISPLAY_WIDTH / original_width
    #     else:
    #         scale_factor = 1.0 # No scaling needed

    #     display_width = int(original_width * scale_factor)
    #     display_height = int(original_height * scale_factor)

    #     # Convert the frame for display and resize it
    #     first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    #     pil_img = Image.fromarray(first_frame_rgb)
    #     display_img = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
    #     st.info(f"Original video resolution: {original_width}x{original_height}. Displayed at: {display_width}x{display_height}")

    #     # Initialize session state to store corners if it doesn't exist
    #     if 'corners' not in st.session_state:
    #         st.session_state.corners = []

    #     # The fragment will manage its own state and updates.
    #     corner_selection_fragment(display_img, scale_factor)

    #     # Display the selected corners so far
    #     if st.session_state.corners:
    #         st.write("Corners selected:")
    #         for i, corner in enumerate(st.session_state.corners):
    #             st.write(f"Corner {i+1}: ({corner[0]}, {corner[1]})")

    #     # Add a button to reset the selection
    #     if st.button("Reset Corners"):
    #         st.session_state.corners = []
    #         st.rerun()

    #     # Check if we have 4 corners selected
    #     if len(st.session_state.corners) == 4:
    #         st.success("4 corners selected! Now upload an image to place inside the frame.")
    #         initial_corners = np.array(st.session_state.corners, dtype=np.float32)

    #         # Let's create two columns for a cleaner layout
    #         col1, col2 = st.columns(2)

    #         with col1:
    #             st.subheader("Step 2: Upload Image")
    #             uploaded_image = st.file_uploader("Choose an image to insert", type=["png", "jpg", "jpeg"])

    #         if uploaded_image:
    #             with col2:
    #                 st.subheader("Step 3: Generate Video")
    #                 st.write("This will track the corners and prepare for the final render.")
    #                 process_button = st.button("Track Points & Generate Video")

    #             if process_button:
    #                 with st.spinner("Tracking corners through video... this may take a while."):
    #                     # --- The Tracking Logic Begins ---

    #                     # 1. Re-open the video to start from the beginning
    #                     vid_capture = cv2.VideoCapture(video_path)
    #                     total_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    #                     # Initialize progress bar and a counter for the frames
    #                     progress_text = "Tracking progress... Initializing."
    #                     progress_bar = st.progress(0, text=progress_text)
    #                     frame_num = 0
                        
    #                     # Read the first frame and convert to gray
    #                     success, first_frame = vid_capture.read()
    #                     prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    #                     # Reshape our initial_corners to the format needed by the tracker
    #                     prev_pts = initial_corners.reshape(-1, 1, 2)

    #                     # Create a list to store the tracked points for each frame
    #                     # We'll start it with the points from the first frame
    #                     tracked_points_history = [prev_pts]
                        
    #                     # 2. Set up parameters for the Lucas-Kanade optical flow
    #                     lk_params = dict(winSize=(15, 15),
    #                                 maxLevel=2,
    #                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        
    #                     # 3. Loop through the video frames
    #                     while True:
    #                         # Update the frame counter and progress bar
    #                         frame_num += 1
    #                         percent_complete = frame_num / total_frames
    #                         progress_bar.progress(percent_complete, text=f"Processing frame {frame_num} of {total_frames}")

    #                         success, frame = vid_capture.read()
    #                         if not success:
    #                             break # End of video
                            
    #                         # Convert the new frame to gray
    #                         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #                         # Calculate optical flow (i.e., track the points)
    #                         next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)

    #                         # Check the status. If all points are tracked successfully, status will contain all 1s.
    #                         if next_pts is not None and np.all(status == 1):
    #                             # Tracking was successful for all points
    #                             tracked_points_history.append(next_pts)
    #                             # For the next iteration, the current frame becomes the previous one
    #                             prev_gray = frame_gray.copy()
    #                             prev_pts = next_pts.reshape(-1, 1, 2)
    #                         else:
    #                             # Tracking failed for at least one point. Stop processing.
    #                             st.warning(f"Corner tracking lost at frame {len(tracked_points_history)}. Processing will stop here.")
    #                             break
                        
    #                     # 4. Release the video capture object
    #                     vid_capture.release()

    #                     # Store the results in session state so the next step can use them
    #                     st.session_state.tracked_points = tracked_points_history
                        
    #                     st.success(f"Successfully tracked corners for {len(tracked_points_history)} frames!")
    #                     st.info("Ready for the final step: Warping the image and rendering the video.")

                    
else:
    st.info("Please upload a video file to begin.")



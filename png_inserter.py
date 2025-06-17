import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import tempfile
import os
import concurrent.futures
import av
from fractions import Fraction
import requests
import urllib.parse
import platform

# Set page layout to wide for a better experience
st.set_page_config(layout="wide")

def is_streamlit_cloud():
    # On Streamlit Cloud, platform.processor() is usually empty
    return platform.processor() == ""

def calculate_image_placement(image_size, target_corners, padding_percent=0.05):
    """
    Calculates the placement of an image inside a target quadrilateral,
    maintaining aspect ratio and adding padding.

    Args:
        image_size (tuple): The (width, height) of the source image.
        target_corners (np.ndarray): The 4x2 array of the destination quadrilateral.
        padding_percent (float): The percentage of padding to add inside the frame.

    Returns:
        np.ndarray: The 4x2 array of the image's calculated destination corners.
    """
    img_w, img_h = image_size
    img_aspect_ratio = img_w / img_h

    # Find the bounding box of the target quadrilateral
    min_x, min_y = np.min(target_corners, axis=0)
    max_x, max_y = np.max(target_corners, axis=0)
    target_bbox_w = max_x - min_x
    target_bbox_h = max_y - min_y

    # Apply padding
    padded_w = target_bbox_w * (1 - 2 * padding_percent)
    padded_h = target_bbox_h * (1 - 2 * padding_percent)
    
    if padded_w <= 0 or padded_h <= 0:
        return None # Avoid division by zero if the target area is too small
        
    target_aspect_ratio = padded_w / padded_h

    # Fit the image into the padded box, maintaining aspect ratio
    if img_aspect_ratio > target_aspect_ratio:
        # Image is wider than the target box, so width is the limiting factor
        final_w = padded_w
        final_h = final_w / img_aspect_ratio
    else:
        # Image is taller, so height is the limiting factor
        final_h = padded_h
        final_w = final_h * img_aspect_ratio

    # Center the final box within the padded area
    offset_x = (padded_w - final_w) / 2
    offset_y = (padded_h - final_h) / 2

    final_x = min_x + (target_bbox_w * padding_percent) + offset_x
    final_y = min_y + (target_bbox_h * padding_percent) + offset_y

    # Define the four corners of the image's destination rectangle
    dest_corners = np.array([
        [final_x, final_y],
        [final_x + final_w, final_y],
        [final_x + final_w, final_y + final_h],
        [final_x, final_y + final_h]
    ], dtype=np.float32)

    return dest_corners

def gdrive_to_direct(url):
    # Extract file ID from Google Drive share URL
    import re
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if not m:
        return url  # fallback: return as-is
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

st.title("Mockup Creator")
if is_streamlit_cloud():
    st.info("Running on Streamlit Cloud")
else:
    st.info("Running locally")
st.write("""
Upload the PNG image you want to insert.
""")

# Create three columns for the file uploaders for a clean layout
col1, col2, col3 = st.columns(3)

# with col1:
#     uploaded_video = st.file_uploader("1. Choose the original video file", type=["mp4", "mov", "avi"])

# with col2:
#     uploaded_json = st.file_uploader("2. Choose the coordinates JSON file", type=["json"])

json_url = gdrive_to_direct("https://drive.google.com/file/d/1lYMB_tMXu-o0DrheDvIKUee7SVWfCTm3/view?usp=sharing")
video_url = gdrive_to_direct("https://drive.google.com/file/d/1ei8BJGEf0EIjSy6ggntJs4Rv86mNeuAx/view?usp=sharing")

with col1:
    uploaded_image = st.file_uploader("3. Choose the PNG/JPG file to insert", type=["png", "jpg", "jpeg"])

if uploaded_image: # Check for the new variable name
    st.success("All files uploaded! Ready to process.")

    if st.button("🚀 Generate Final Video", type="primary"):
        # --- 1. PRE-PROCESSING AND SETUP ---
        with st.spinner("Preparing assets..."):
            # Load data
            # video_bytes = uploaded_video.getvalue()
            # coords_data = json.load(uploaded_json)

            # Download video to a temp file
            video_response = requests.get(video_url, stream=True)
            video_response.raise_for_status()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in video_response.iter_content(chunk_size=8192):
                tfile.write(chunk)
            tfile.flush()
            tfile.seek(0)
            video_bytes = tfile.read()
            tfile.seek(0)

            # Download JSON
            json_response = requests.get(json_url)
            json_response.raise_for_status()
            coords_data = json_response.json()

            # --- Simplified Image Handling ---
            # Open the uploaded PNG/JPG with Pillow and ensure it has an alpha channel (RGBA)
            pil_image = Image.open(uploaded_image).convert("RGBA")
            # --- Crop transparent border ---
            np_img = np.array(pil_image)
            alpha = np_img[:, :, 3]
            # You can adjust the threshold (e.g., >10 for "nearly" transparent)
            mask = alpha > 10
            coords = np.argwhere(mask)
            if coords.size > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
                pil_image = pil_image.crop((x0, y0, x1, y1))
            # Convert to an OpenCV image (BGRA for compositing)
            image_to_insert = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
            image_to_insert = cv2.GaussianBlur(image_to_insert, (3, 3), 0)
            desired_alpha = 0.85  # Change this value between 0 (fully transparent) and 1 (fully opaque)
            image_to_insert[:, :, 3] = (image_to_insert[:, :, 3].astype(np.float32) * desired_alpha).astype(np.uint8)
            img_h, img_w, _ = image_to_insert.shape
            # These are the corners of our flat source image, a simple rectangle
            image_src_corners = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
            
            # Get video properties
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_bytes)
            vid_capture = cv2.VideoCapture(tfile.name)
            total_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_w = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vid_capture.get(cv2.CAP_PROP_FPS)
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Setup video writer for output
            # output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            video_base = os.path.splitext(os.path.basename(urllib.parse.urlparse(video_url).path))[0]
            image_base = os.path.splitext(uploaded_image.name)[0]
            output_filename = f"{video_base}__{image_base}.mp4"
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            # fourcc = cv2.VideoWriter_fourcc(*'VP80')
            # out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_w, video_h))

        # --- 2. FRAME-BY-FRAME PROCESSING (PARALLELIZED) ---

        def process_frame(args):
            """
            Processes a single frame.
            If coordinates are found, it blends the image.
            If not, it returns the original, unmodified frame.
            """
            frame_num, frame = args
            frame_key = str(frame_num)

            if frame_key in coords_data:
                tracked_corners = np.array(coords_data[frame_key], dtype=np.float32)
                
                # Ensure img_w and img_h are available. You may need to pass them as args.
                # For this example, assuming they are accessible in the scope.
                dest_corners = calculate_image_placement((img_w, img_h), tracked_corners, padding_percent=0.05)
                
                if dest_corners is not None:
                    M = cv2.getPerspectiveTransform(image_src_corners, dest_corners)
                    warped_image = cv2.warpPerspective(
                        image_to_insert, M, (video_w, video_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT
                    )
                    
                    # Perform blending
                    alpha = warped_image[:, :, 3:4].astype(np.float32) / 255.0
                    inv_alpha = 1.0 - alpha
                    
                    # Create the blended frame
                    blended_frame = (frame.astype(np.float32) * inv_alpha + warped_image[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
                    
                    return frame_num, blended_frame

            # If coords_data[frame_key] doesn't exist or dest_corners is None,
            # return the original, untouched frame.
            return frame_num, frame
        
        # 1. Read all frames into memory (fast, avoids thread-unsafe VideoCapture)
        if not is_streamlit_cloud():
            progress_bar = st.progress(0, "Processing video...")
            frames = []
            for frame_num in range(total_frames):
                success, frame = vid_capture.read()
                if not success:
                    break
                frames.append(frame)

            # 2. Process frames in parallel
            processed_frames = [None] * len(frames)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_frame, (i, frames[i])): i for i in range(len(frames))}
                for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    frame_num, processed = future.result()
                    processed_frames[frame_num] = processed
                    progress_bar.progress((idx + 1) / len(frames), f"Processing frame {idx + 1}/{len(frames)}")

            # 3. Write frames in order using PyAV
            progress_bar = st.progress(0, "Encoding video with PyAV...")
            container = av.open(output_video_path, mode='w')
            stream = container.add_stream('libx264', rate=Fraction(fps).limit_denominator())
            stream.width = video_w
            stream.height = video_h
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18'}  # Lower CRF = higher quality (try 4-10 for best results)

            for idx, frame in enumerate(processed_frames):
                # Convert BGR (OpenCV) to RGB for PyAV
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                if idx % 5 == 0 or idx == len(processed_frames) - 1:
                    progress_bar.progress((idx + 1) / len(processed_frames), f"Encoding frame {idx + 1}/{len(processed_frames)}")

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        else:
            # Cloud: Process and write each frame sequentially to save memory
            progress_bar = st.progress(0, "Processing and encoding video (cloud mode)...")
            # container = av.open(output_video_path, mode='w')
            # stream = container.add_stream('libx264', rate=Fraction(fps).limit_denominator())
            # stream.width = video_w
            # stream.height = video_h
            # stream.pix_fmt = 'yuv420p'
            # stream.options = {'crf': '18', 'g': '1'}
            # --- NEW CODE: Use the robust MJPEG codec for debugging ---
            container = av.open(output_video_path, mode='w')
            stream = container.add_stream('mjpeg', rate=Fraction(fps).limit_denominator())

            stream.width = video_w
            stream.height = video_h

            # MJPEG uses a different pixel format and quality setting
            stream.pix_fmt = 'yuvj420p' 
            stream.options = {'q:v': '2'} # Quality setting, 2-5 is high quality

            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for idx in range(total_frames):
                success, original_frame = vid_capture.read()
                if not success or original_frame is None:
                    st.warning(f"⚠️ Could not read frame {idx}. Skipping.")
                    continue

                # Create a defensive copy to guarantee it's a new, isolated object in memory.
                frame_to_process = original_frame.copy()

                # Pass the pristine copy to the processing function
                _, processed_frame = process_frame((idx, frame_to_process))

                if processed_frame is None:
                    st.warning(f"⚠️ Processing failed for frame {idx}. Re-using original frame.")
                    processed_frame = original_frame # Fallback to the original frame to prevent flicker/gaps

                # --- Robust Conversion for PyAV ---
                # 1. Ensure the processed frame is uint8
                if processed_frame.dtype != np.uint8:
                    processed_frame = processed_frame.clip(0, 255).astype(np.uint8)

                # 2. Ensure it has 3 channels (it should after process_frame)
                if processed_frame.ndim != 3 or processed_frame.shape[2] != 3:
                    st.warning(f"⚠️ Processed frame {idx} has wrong dimensions {processed_frame.shape}. Skipping.")
                    continue

                # 3. **Crucially, convert the final BGR frame to RGB for PyAV**
                try:
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    st.error(f"Error converting frame {idx} to RGB: {e}")
                    continue

                # Ensure the frame data is in a C-contiguous block of memory for PyAV.
                contiguous_frame_rgb = np.ascontiguousarray(frame_rgb)

                # Double-check shape before writing
                if contiguous_frame_rgb.shape != (video_h, video_w, 3):
                    print(f"Skipping frame {idx}: shape mismatch {contiguous_frame_rgb.shape} vs expected {(video_h, video_w, 3)}")
                    continue

                # Encode the frame
                av_frame = av.VideoFrame.from_ndarray(contiguous_frame_rgb, format='rgb24')
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                if idx % 5 == 0 or idx == total_frames - 1:
                    progress_bar.progress((idx + 1) / total_frames, f"Processing frame {idx + 1}/{total_frames}")

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            container.close()

        # --- 3. CLEANUP AND DISPLAY ---
        vid_capture.release()
        tfile.close()
        os.unlink(tfile.name)

        st.success("Video generation complete!")
        c = st.columns(2)
        with c[0]:
            st.video(output_video_path)
        
        with open(output_video_path, "rb") as f:
            st.download_button("⬇️ Download Video", f, output_filename, "video/mp4")

else:
    st.info("Please upload a file to begin.")
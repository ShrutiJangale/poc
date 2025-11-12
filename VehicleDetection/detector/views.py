import os
import cv2
import json
from collections import defaultdict
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort # Assuming this is your DeepSort implementation
from .forms import VideoUploadForm

# --- Configuration ---
MODEL = YOLO("yolov8n.pt") # Your YOLO model
CLASSES_TO_COUNT = {'car', 'truck', 'bus'} # Classes you want to count

# --- In-memory cache for analysis results ---
# Note: For production with multiple workers, use a proper cache like Redis.
ANALYSIS_DATA = {}

def home(request):
    """
    Handles the main page and video upload.
    """
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_instance = form.save()
            return JsonResponse({'success': True, 'video_url': video_instance.video.url})
        else:
            return JsonResponse({'success': False, 'errors': form.errors})
    
    return render(request, 'detector/index.html')




# Assume MODEL, ANALYSIS_DATA, CLASSES_TO_COUNT are defined elsewhere in your code
# For example:
# from ultralytics import YOLO
# MODEL = YOLO('yolov8n.pt')
# ANALYSIS_DATA = {}
# CLASSES_TO_COUNT = {'car', 'truck', 'bus', 'motorcycle'}

def stream_video_feed(video_path):
    """
    Processes a video stream for robust vehicle detection and counting.
    """
    # Using a slightly higher max_age can help with tracking stability if vehicles
    # are temporarily occluded. n_init helps confirm a track before it's active.
    tracker = DeepSort(max_age=30, n_init=3)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        if video_path in ANALYSIS_DATA:
            ANALYSIS_DATA[video_path]["status"] = "failed"
        return

    # Define the counting line's geometry
    LINE_Y = 550
    LINE_X_START, LINE_X_END = 600, 1200

    # --- Data Structures for Robust Counting ---
    counts = defaultdict(int)
    counted_ids = set()  
    track_history = {}       
    class_memory = {}

    try:
        ANALYSIS_DATA[video_path] = {"status": "processing", "counts": {}}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            # Run YOLOv8 object detection
            results = MODEL.predict(frame, conf=0.5, iou=0.5, classes=[2, 3, 5, 7], verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                class_name = MODEL.names[int(box.cls)]
                if class_name in CLASSES_TO_COUNT:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    # DeepSort expects format: ([x_min, y_min, width, height], confidence, class_name)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_name))

            # Update tracker with new detections
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Draw the counting line on the frame
            cv2.line(frame, (LINE_X_START, LINE_Y), (LINE_X_END, LINE_Y), (0, 255, 0), 3)

            for trk in tracks:
                # We only process confirmed tracks that have been updated in the current frame
                if not trk.is_confirmed() or trk.time_since_update > 0:
                    continue

                tid = trk.track_id
                x1, y1, x2, y2 = map(int, trk.to_tlbr())

                # Lock in the class for the track ID to prevent flickering classifications
                if tid not in class_memory:
                    class_memory[tid] = trk.get_det_class()
                cls = class_memory[tid]
                
                # --- ROBUST COUNTING LOGIC ---

                # 1. Define the counting point: bottom-center of the bounding box.
                # This ensures the vehicle is counted only after it has fully passed the line.
                counting_point_y = y2
                
                # 2. Store the history of the vertical position of the counting point.
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append(counting_point_y)
                # Keep history list from growing indefinitely
                if len(track_history[tid]) > 30:
                    track_history[tid].pop(0)

                # 3. Check for a valid crossing event (requires at least 2 historical points).
                if len(track_history.get(tid, [])) >= 2:
                    prev_y = track_history[tid][-2]
                    current_y = track_history[tid][-1]

                    # Condition A: Vehicle must be uncounted so far.
                    not_counted_yet = tid not in counted_ids
                    
                    # Condition B: Vehicle must have just crossed the line (moving downwards).
                    # This creates a strict "before -> after" rule.
                    crossed_line = prev_y < LINE_Y and current_y >= LINE_Y
                    
                    # Condition C: Vehicle's bounding box must overlap with the line horizontally.
                    # This is more robust than checking just the center point.
                    within_horizontal_zone = x1 < LINE_X_END and x2 > LINE_X_START

                    if not_counted_yet and crossed_line and within_horizontal_zone:
                        counts[cls] += 1
                        counted_ids.add(tid)
                        ANALYSIS_DATA[video_path]["counts"] = dict(counts)
                        print(f"COUNTED: A '{cls}' (ID: {tid}) crossed the line.")
                        # Optional: Change color of the counted vehicle's box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box for counted

                # --- Visualization ---
                # Draw bounding box and track ID
                if tid not in counted_ids:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 128, 0), 2) # Blue box for uncounted
                
                label = f"{cls}-{tid}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 128, 0), 2)

            # Display the running counts on the frame
            y_offset = 40
            for i, (cls, count) in enumerate(counts.items()):
                text = f"{cls.capitalize()}: {count}"
                cv2.putText(frame, text, (800, y_offset + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 20, 120), 3, cv2.LINE_AA)
            
            # Encode frame to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        print(f"Stream for {video_path} finished or was closed.")
        if video_path in ANALYSIS_DATA:
            ANALYSIS_DATA[video_path]["status"] = "completed"
        cap.release()


def video_analysis_stream(request):
    """
    View that returns the streaming video feed.
    """
    video_url = request.GET.get('video_url', None)
    if not video_url:
        return JsonResponse({'error': 'No video URL provided.'}, status=400)

    # Construct the absolute file path to the video
    # video_url comes from Model.video.url, e.g., '/media/videos/my_video.mp4'
    # settings.MEDIA_ROOT is the absolute path to your media directory
    video_file_path = os.path.join(settings.MEDIA_ROOT, video_url.replace('/media/', ''))
    
    if not os.path.exists(video_file_path):
        return JsonResponse({'error': f'Video file not found at {video_file_path}.'}, status=404)

    # Use StreamingHttpResponse to stream the video frames
    return StreamingHttpResponse(
        stream_video_feed(video_file_path),
        content_type='multipart/x-mixed-replace; boundary=frame' # Standard for MJPEG streams
    )


def get_analysis_counts(request):
    """
    API endpoint to fetch the analysis counts for a given video.
    """
    video_url = request.GET.get('video_url', None)
    if not video_url:
        return JsonResponse({'error': 'No video URL provided.'}, status=400)

    video_file_path = os.path.join(settings.MEDIA_ROOT, video_url.replace('/media/', ''))
    
    results = ANALYSIS_DATA.get(video_file_path, {})
    return JsonResponse(results)
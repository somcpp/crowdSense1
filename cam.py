from ultralytics import YOLO
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Global video source flag and path
using_webcam = True
video_path = "uploads/videoplayback (1).mp4"

# Initialize webcam
cap = cv2.VideoCapture(0)

# Heatmap configuration
GRID_ROWS = 8
GRID_COLS = 8
grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
SMOOTHING_FACTOR = 0.3  # Lower values = more smoothing
smoothed_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
total_people_count = 0

# Class index for 'person' in COCO dataset (used by YOLOv8)
PERSON_CLASS_ID = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    global grid_counts, smoothed_grid, total_people_count, cap, using_webcam, video_path
    
    last_update_time = time.time()
    
    # If video file was uploaded and we're not using webcam
    if not using_webcam and video_path:
        # Release current video source if any
        if cap is not None:
            cap.release()
        # Open the uploaded video file
        cap = cv2.VideoCapture(video_path)
    
    # Make sure we have a valid video source
    if not cap.isOpened():
        if using_webcam:
            # Try to open the webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + 
                       b'\r\n')
                return
        else:
            # Invalid video file
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + 
                   b'\r\n')
            return
    
    while True:
        success, frame = cap.read()
        
        # If end of video, loop back to beginning for uploaded videos
        if not success:
            if not using_webcam and video_path:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
                success, frame = cap.read()
                if not success:
                    break
            else:
                # For webcam, just try to get the next frame
                continue
            
        height, width, _ = frame.shape
        cell_h = height // GRID_ROWS
        cell_w = width // GRID_COLS
        
        # Limit updates to reduce flickering (process every 100ms)
        current_time = time.time()
        if current_time - last_update_time > 0.1:
            # Reset grid counts for each frame
            grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
            
            # Detect objects
            results = model(frame)
            
            # Track people count for this frame
            frame_people_count = 0
            
            # Process detected objects - only count people
            if results and len(results) > 0:
                for i, box in enumerate(results[0].boxes):
                    # Only process if it's a person
                    if int(box.cls.item()) == PERSON_CLASS_ID:
                        frame_people_count += 1
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate center of the detected person
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        # Determine which grid cell the center falls into
                        row = min(cy // cell_h, GRID_ROWS - 1)
                        col = min(cx // cell_w, GRID_COLS - 1)
                        
                        # Increment count for that cell
                        grid_counts[row][col] += 1
                        
                        # Draw a small circle at the center of the detected person
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Label the person
                        cv2.putText(frame, f"Person {i+1}", (int(x1), int(y1) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update total people count with the current frame count
            total_people_count = frame_people_count
            
            # Update smoothed grid with new counts (exponential moving average)
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    smoothed_grid[r][c] = (SMOOTHING_FACTOR * grid_counts[r][c] + 
                                          (1 - SMOOTHING_FACTOR) * smoothed_grid[r][c])
            
            last_update_time = current_time
        
        # Add people count to the frame
        cv2.putText(frame, f"People Count: {total_people_count}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add source indicator
        source_text = "Source: Webcam" if using_webcam else f"Source: Video - {os.path.basename(video_path)}"
        cv2.putText(frame, source_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, source_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Convert the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add a small delay to control frame rate for uploaded videos
        if not using_webcam:
            time.sleep(0.03)  # ~30fps

def generate_heatmap():
    global smoothed_grid
    while True:
        time.sleep(0.1)  # Slow down updates to reduce flickering
        
        # Create a heatmap visualization with consistent size
        heatmap_width = 400
        heatmap_height = 400
        heatmap = np.zeros((heatmap_height, heatmap_width, 3), dtype=np.uint8)
        
        # Calculate cell dimensions (ensure they're integers)
        cell_h = heatmap_height // GRID_ROWS
        cell_w = heatmap_width // GRID_COLS
        
        # Find the maximum value for normalization
        max_count = max(max(row) for row in smoothed_grid) if any(any(row) for row in smoothed_grid) else 1
        if max_count < 0.1:  # Avoid division by near-zero
            max_count = 0.1
            
        # Fill the entire heatmap with green (base color when no people)
        heatmap[:, :] = (0, 180, 0)  # Green background
        
        # Draw the heatmap cells
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Normalize count to get color intensity (0-1)
                intensity = min(1.0, smoothed_grid[row][col] / max_count)
                
                if intensity > 0.01:  # Only color cells with some activity
                    # Create a smooth gradient from green (no people) to red (many people)
                    # BGR: Green (0,180,0) → Yellow (0,180,180) → Orange (0,100,200) → Red (0,0,255)
                    if intensity < 0.3:
                        # Green to Yellow
                        b_value = int(180 * (intensity / 0.3))
                        color = (b_value, 180, 0)  # BGR format
                    elif intensity < 0.6:
                        # Yellow to Orange
                        adjusted = (intensity - 0.3) / 0.3
                        g_value = int(180 - 80 * adjusted)
                        b_value = int(180 + (200 - 180) * adjusted)
                        color = (b_value, g_value, 0)  # BGR format
                    else:
                        # Orange to Red
                        adjusted = (intensity - 0.6) / 0.4
                        g_value = int(100 - 100 * adjusted)
                        color = (255, g_value, 0)  # BGR format
                    
                    # Calculate the exact pixel coordinates for this cell
                    y1 = row * cell_h
                    y2 = (row + 1) * cell_h
                    x1 = col * cell_w
                    x2 = (col + 1) * cell_w
                    
                    # Fill the cell with the color
                    cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, -1)
        
        # Draw consistent, subtle grid lines as a separate overlay
        grid_color = (70, 70, 70)  # Dark gray, slightly visible
        grid_alpha = 0.15  # Very transparent
        
        # Create a grid overlay
        grid_overlay = np.zeros_like(heatmap, dtype=np.float32)
        
        # Draw horizontal grid lines
        for i in range(1, GRID_ROWS):
            y = i * cell_h
            cv2.line(grid_overlay, (0, y), (heatmap_width, y), (1, 1, 1), 1)
            
        # Draw vertical grid lines
        for i in range(1, GRID_COLS):
            x = i * cell_w
            cv2.line(grid_overlay, (x, 0), (x, heatmap_height), (1, 1, 1), 1)
            
        # Apply the grid overlay with transparency
        heatmap = cv2.addWeighted(heatmap, 1, grid_overlay.astype(np.uint8) * 255, grid_alpha, 0)
        
        # Add a title with the total people count
        cv2.putText(heatmap, f"Person Activity Heatmap", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        legend_y = heatmap_height - 60
        # Green (low)
        cv2.rectangle(heatmap, (10, legend_y), (30, legend_y + 20), (0, 180, 0), -1)
        cv2.putText(heatmap, "Low", (35, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Yellow (medium)
        cv2.rectangle(heatmap, (80, legend_y), (100, legend_y + 20), (180, 180, 0), -1)
        cv2.putText(heatmap, "Medium", (105, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Red (high)
        cv2.rectangle(heatmap, (180, legend_y), (200, legend_y + 20), (255, 0, 0), -1)
        cv2.putText(heatmap, "High", (205, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', heatmap)
        heatmap_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + heatmap_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html', GRID_ROWS=GRID_ROWS, GRID_COLS=GRID_COLS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap_feed')
def heatmap_feed():
    return Response(generate_heatmap(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap_data')
def heatmap_data():
    return jsonify({
        'grid': smoothed_grid,
        'people_count': total_people_count
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    global using_webcam, video_path
    
    # Check if a file was submitted
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    
    # If user submits empty form
    if file.filename == '':
        return redirect(request.url)
    
    # If valid file and allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Remove old uploaded files to save space (optional)
        for old_file in os.listdir(app.config['UPLOAD_FOLDER']):
            old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_file)
            if os.path.isfile(old_path):
                os.unlink(old_path)
        
        # Save the new file
        file.save(file_path)
        
        # Set the video path and flag
        video_path = file_path
        using_webcam = False
        
        return redirect(url_for('home'))
    
    return redirect(request.url)

@app.route('/use_webcam')
def use_webcam():
    global using_webcam, cap
    
    # Switch to webcam mode
    using_webcam = True
    
    # Release current video if not webcam
    if cap is not None:
        cap.release()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
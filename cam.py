from ultralytics import YOLO
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import time

app = Flask(__name__)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Increase grid resolution for better heatmap visualization
GRID_ROWS = 8
GRID_COLS = 8
grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

# For stability, we'll use a moving average
SMOOTHING_FACTOR = 0.3  # Lower values = more smoothing
smoothed_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
total_people_count = 0

# Class index for 'person' in COCO dataset (used by YOLOv8)
PERSON_CLASS_ID = 0

def generate_frames():
    global grid_counts, smoothed_grid, total_people_count
    last_update_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
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
        
        # Convert the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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

if __name__ == "__main__":
    app.run(debug=True)
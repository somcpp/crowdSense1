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

# Heatmap configuration - increased resolution for smoother gradients
GRID_ROWS = 32  # Increased from 8 for smoother gradient
GRID_COLS = 32  # Increased from 8 for smoother gradient
grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
SMOOTHING_FACTOR = 0.3  # Lower values = more smoothing
smoothed_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
total_people_count = 0

# Crowd density thresholds (adjust based on your needs)
LOW_THRESHOLD = 8
MEDIUM_THRESHOLD = 12
HIGH_THRESHOLD = 14

# Class index for 'person' in COCO dataset (used by YOLOv8)
PERSON_CLASS_ID = 0

# Zone definitions - define zones as rectangles [x1, y1, x2, y2]
# Coordinates are normalized (0.0-1.0) to be resolution-independent
zones = {
    'A': {'coords': [0.0, 0.0, 0.33, 1.0], 'color': (0, 0, 255), 'count': 0, 'density': 'low'},  # Left zone (red)
    'B': {'coords': [0.33, 0.0, 0.66, 1.0], 'color': (0, 255, 0), 'count': 0, 'density': 'low'},  # Middle zone (green)
    'C': {'coords': [0.66, 0.0, 1.0, 1.0], 'color': (255, 0, 0), 'count': 0, 'density': 'low'}   # Right zone (blue)
}

# To store movement recommendations
movement_recommendations = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_zone_densities():
    """Calculate crowd density for each zone and generate movement recommendations"""
    global zones, movement_recommendations
    
    # Reset recommendations
    movement_recommendations = []
    
    # First determine density levels for each zone
    for zone_id, zone in zones.items():
        people_count = zone['count']
        if people_count < LOW_THRESHOLD:
            zone['density'] = 'low'
        elif people_count < MEDIUM_THRESHOLD:
            zone['density'] = 'medium'
        else:
            zone['density'] = 'high'
    
    # Generate movement recommendations based on density differences
    high_density_zones = []
    low_density_zones = []
    
    for zone_id, zone in zones.items():
        if zone['density'] == 'high':
            high_density_zones.append(zone_id)
        elif zone['density'] == 'low':
            low_density_zones.append(zone_id)
    
    # Generate recommendations from high to low density zones
    for high_zone in high_density_zones:
        for low_zone in low_density_zones:
            if high_zone != low_zone:
                recommendation = f"Move from Zone {high_zone} to Zone {low_zone}"
                movement_recommendations.append(recommendation)
    
    # If no clear recommendations (all zones similar density)
    if not movement_recommendations:
        if all(zone['density'] == 'high' for zone in zones.values()):
            movement_recommendations.append("All zones at capacity. Consider opening new areas.")
        elif all(zone['density'] == 'low' for zone in zones.values()):
            movement_recommendations.append("All zones have low occupancy. No movement needed.")
        else:
            movement_recommendations.append("Crowd distribution is balanced across zones.")

def generate_frames():
    global grid_counts, smoothed_grid, total_people_count, cap, using_webcam, video_path, zones
    
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
            # Reset grid counts and zone counts for each frame
            grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
            for zone_id in zones:
                zones[zone_id]['count'] = 0
            
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
                        
                        # Check which zone the person is in
                        person_added_to_zone = False
                        for zone_id, zone_info in zones.items():
                            # Convert normalized coordinates to actual pixel values
                            zone_x1 = int(zone_info['coords'][0] * width)
                            zone_y1 = int(zone_info['coords'][1] * height)
                            zone_x2 = int(zone_info['coords'][2] * width)
                            zone_y2 = int(zone_info['coords'][3] * height)
                            
                            # Check if person center is within this zone
                            if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
                                zones[zone_id]['count'] += 1
                                person_added_to_zone = True
                        
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
            
            # Calculate crowd density for each zone
            calculate_zone_densities()
            
            # Update smoothed grid with new counts (exponential moving average)
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    smoothed_grid[r][c] = (SMOOTHING_FACTOR * grid_counts[r][c] + 
                                          (1 - SMOOTHING_FACTOR) * smoothed_grid[r][c])
            
            last_update_time = current_time
        
        # Add people count to the frame
        cv2.putText(frame, f"Total People: {total_people_count}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw zone boundaries and add count labels
        for zone_id, zone_info in zones.items():
            # Convert normalized coordinates to actual pixel values
            x1 = int(zone_info['coords'][0] * width)
            y1 = int(zone_info['coords'][1] * height)
            x2 = int(zone_info['coords'][2] * width)
            y2 = int(zone_info['coords'][3] * height)
            
            # Get density-based color (red for high, yellow for medium, green for low)
            if zone_info['density'] == 'high':
                density_color = (0, 0, 255)  # Red (BGR)
            elif zone_info['density'] == 'medium':
                density_color = (0, 165, 255)  # Orange (BGR)
            else:
                density_color = (0, 255, 0)  # Green (BGR)
            
            # Draw translucent zone overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_info['color'], -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # Make overlay translucent
            
            # Draw zone border
            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_info['color'], 2)
            
            # Add zone label and count
            label_bg_y1 = y1 + 5
            label_bg_y2 = y1 + 65
            label_bg_x1 = x1 + 5
            label_bg_x2 = x1 + 140
            
            # Semi-transparent background for label
            overlay = frame.copy()
            cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Add text
            cv2.putText(frame, f"Zone {zone_id}", (label_bg_x1 + 5, label_bg_y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Count: {zone_info['count']}", (label_bg_x1 + 5, label_bg_y1 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"{zone_info['density'].capitalize()}", (label_bg_x1 + 5, label_bg_y1 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 2)
        
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
    global smoothed_grid, total_people_count, zones, movement_recommendations
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
            
        # Fill the entire heatmap with dark blue (base color when no people)
        heatmap[:, :] = (100, 0, 0)  # Dark blue background (BGR)
        
        # Create a blank base for the heatmap
        base_heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
        
        # Fill the base heatmap with intensity values
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if smoothed_grid[row][col] > 0.01:
                    # Determine area to affect with Gaussian distribution (smoother effect)
                    y_center = int((row + 0.5) * cell_h)
                    x_center = int((col + 0.5) * cell_w)
                    
                    # Intensity based on normalized value
                    intensity = min(1.0, smoothed_grid[row][col] / max_count)
                    
                    # Apply Gaussian blob to create smooth transitions (Weather-map like effect)
                    # The sigma controls the spread - larger values create more spread
                    sigma = max(cell_h, cell_w) * 0.5  # Adjust for desired spread
                    
                    # Create a weighted Gaussian blob
                    y, x = np.ogrid[-y_center:heatmap_height-y_center, -x_center:heatmap_width-x_center]
                    mask = np.exp(-(x*x + y*y) / (2*sigma*sigma))
                    
                    # Apply the weighted mask to the base heatmap
                    base_heatmap = np.maximum(base_heatmap, mask * intensity)
        
        # Apply colormap to the base heatmap
        # Map values from 0-1 to appropriate colors (blue to green to yellow to red)
        for y in range(heatmap_height):
            for x in range(heatmap_width):
                intensity = base_heatmap[y, x]
                if intensity > 0.01:  # Only color cells with some activity
                    # Create a smooth gradient from blue (low) to green to yellow to red (high)
                    # BGR format
                    if intensity < 0.25:
                        # Blue to Cyan
                        g_value = int(255 * (intensity / 0.25))
                        color = (255, g_value, 0)  # BGR
                    elif intensity < 0.5:
                        # Cyan to Green
                        b_value = int(255 - 255 * ((intensity - 0.25) / 0.25))
                        color = (b_value, 255, 0)  # BGR
                    elif intensity < 0.75:
                        # Green to Yellow
                        r_value = int(255 * ((intensity - 0.5) / 0.25))
                        color = (0, 255, r_value)  # BGR
                    else:
                        # Yellow to Red
                        g_value = int(255 - 255 * ((intensity - 0.75) / 0.25))
                        color = (0, g_value, 255)  # BGR
                    
                    # Apply color to the heatmap
                    heatmap[y, x] = color
        
        # Apply mild Gaussian blur for even smoother transitions
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # Add a title
        cv2.putText(heatmap, f"Activity Heatmap", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw zone boundaries on heatmap
        for zone_id, zone_info in zones.items():
            # Convert normalized coordinates to actual pixel values
            x1 = int(zone_info['coords'][0] * heatmap_width)
            y1 = int(zone_info['coords'][1] * heatmap_height)
            x2 = int(zone_info['coords'][2] * heatmap_width)
            y2 = int(zone_info['coords'][3] * heatmap_height)
            
            # Draw zone border
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Add zone label
            cv2.putText(heatmap, f"Zone {zone_id}", (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Determine crowd density level and color for count box
        if total_people_count < LOW_THRESHOLD:
            density_text = "Low Overall Density"
            count_color = (0, 255, 0)  # Green for low (BGR)
        elif total_people_count < MEDIUM_THRESHOLD:
            density_text = "Medium Overall Density"
            count_color = (0, 165, 255)  # Orange for medium (BGR)
        else:
            density_text = "High Overall Density"
            count_color = (0, 0, 255)  # Red for high (BGR)
            
        # Add a box with people count that changes color based on density
        count_box_y1 = heatmap_height - 80
        count_box_y2 = heatmap_height - 20
        count_box_x1 = heatmap_width // 2 - 100
        count_box_x2 = heatmap_width // 2 + 100
        
        # Draw semi-transparent background
        overlay = heatmap.copy()
        cv2.rectangle(overlay, (count_box_x1, count_box_y1), (count_box_x2, count_box_y2), count_color, -1)
        cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0, heatmap)
        
        # Draw border
        cv2.rectangle(heatmap, (count_box_x1, count_box_y1), (count_box_x2, count_box_y2), (255, 255, 255), 2)
        
        # Add count text
        cv2.putText(heatmap, f"Total People: {total_people_count}", 
                   (count_box_x1 + 10, count_box_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(heatmap, density_text, 
                   (count_box_x1 + 10, count_box_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add a small gradient legend
        legend_height = 20
        legend_width = 200
        legend_x = 20
        legend_y = heatmap_height - 120
        
        # Create gradient for legend
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        for x in range(legend_width):
            ratio = x / legend_width
            if ratio < 0.25:
                # Blue to Cyan
                g_value = int(255 * (ratio / 0.25))
                color = (255, g_value, 0)  # BGR
            elif ratio < 0.5:
                # Cyan to Green
                b_value = int(255 - 255 * ((ratio - 0.25) / 0.25))
                color = (b_value, 255, 0)  # BGR
            elif ratio < 0.75:
                # Green to Yellow
                r_value = int(255 * ((ratio - 0.5) / 0.25))
                color = (0, 255, r_value)  # BGR
            else:
                # Yellow to Red
                g_value = int(255 - 255 * ((ratio - 0.75) / 0.25))
                color = (0, g_value, 255)  # BGR
            
            # Fill the column with the color
            legend[:, x] = color
        
        # Add the legend to the heatmap
        heatmap[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width] = legend
        
        # Legend labels
        cv2.putText(heatmap, "Low", (legend_x, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(heatmap, "High", (legend_x + legend_width - 30, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add movement recommendations box
        if movement_recommendations:
            recommend_box_y1 = 60
            recommend_box_x1 = 20
            recommend_box_width = heatmap_width - 40
            recommend_box_height = 30 + (len(movement_recommendations) * 25)
            recommend_box_y2 = recommend_box_y1 + recommend_box_height
            recommend_box_x2 = recommend_box_x1 + recommend_box_width
            
            # Draw semi-transparent background
            overlay = heatmap.copy()
            cv2.rectangle(overlay, (recommend_box_x1, recommend_box_y1), 
                         (recommend_box_x2, recommend_box_y2), (0, 0, 128), -1)
            cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0, heatmap)
            
            # Draw border
            cv2.rectangle(heatmap, (recommend_box_x1, recommend_box_y1), 
                         (recommend_box_x2, recommend_box_y2), (255, 255, 255), 2)
            
            # Add title
            cv2.putText(heatmap, "Movement Recommendations:", 
                       (recommend_box_x1 + 10, recommend_box_y1 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add recommendations
            for i, recommendation in enumerate(movement_recommendations):
                cv2.putText(heatmap, recommendation, 
                           (recommend_box_x1 + 10, recommend_box_y1 + 50 + (i * 25)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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

@app.route('/zone_data')
def zone_data():
    """Return zone-specific data as JSON"""
    return jsonify({
        'zones': zones,
        'total_people': total_people_count,
        'recommendations': movement_recommendations
    })

@app.route('/heatmap_data')
def heatmap_data():
    # Determine crowd density level
    if total_people_count < LOW_THRESHOLD:
        density_level = "low"
    elif total_people_count < MEDIUM_THRESHOLD:
        density_level = "medium"
    else:
        density_level = "high"
    
    return jsonify({
        'grid': smoothed_grid,
        'people_count': total_people_count,
        'density_level': density_level
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

@app.route('/update_zones', methods=['POST'])
def update_zones():
    """Update zone configuration"""
    global zones
    
    data = request.json
    if data and 'zones' in data:
        new_zones = data['zones']
        # Validate and update zone configuration
        for zone_id, zone_info in new_zones.items():
            if zone_id in zones and 'coords' in zone_info:
                # Update coordinates if provided
                if all(0 <= x <= 1 for x in zone_info['coords']):
                    zones[zone_id]['coords'] = zone_info['coords']
                
                # Update color if provided
                if 'color' in zone_info and len(zone_info['color']) == 3:
                    zones[zone_id]['color'] = zone_info['color']
    
    return jsonify({'success': True, 'zones': zones})

if __name__ == "__main__":
    app.run(debug=True)
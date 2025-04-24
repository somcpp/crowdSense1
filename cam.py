from ultralytics import YOLO
import cv2
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

GRID_ROWS = 4
GRID_COLS = 4
grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

def generate_frames():
    global grid_counts
    while True:
        success, frame = cap.read()
        if not success:
            continue

        height, width, _ = frame.shape
        cell_h = height // GRID_ROWS
        cell_w = width // GRID_COLS

        grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

        results = model(frame)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            row = min(cy // cell_h, GRID_ROWS - 1)
            col = min(cx // cell_w, GRID_COLS - 1)
            grid_counts[row][col] += 1

            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap_data')
def heatmap_data():
    return jsonify({'grid': grid_counts})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import os
import subprocess
import yaml

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to YOLOv5 directory and detect.py
yolov5_dir = r'C:\Users\Abin\Desktop\crop_prediction\yolov5'
detect_script_path = os.path.join(yolov5_dir, 'detect.py')

# Load class names from dataset.yaml
with open(r'C:\Users\Abin\Desktop\crop_prediction\yolov5\dataset.yaml', 'r') as file:
    dataset = yaml.safe_load(file)
    class_names = dataset['names']  # This should be a list of class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Save the uploaded image to the 'uploaded' folder
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        abs_image_path = os.path.abspath(image_path)

        # YOLOv5 parameters
        weights_path = r'C:\Users\Abin\Desktop\crop_prediction\yolov5\runs\train\exp2\weights\best.pt'
        output_dir = r'C:\Users\Abin\Desktop\crop_prediction\yolov5\runs\detect'

        # Command to run YOLOv5 detection
        cmd = [
            'python', detect_script_path,
            '--weights', weights_path,
            '--img', '416',
            '--conf', '0.15',
            '--source', abs_image_path,
            '--project', output_dir,
            '--name', 'exp',
            '--save-txt', '--save-conf'
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=yolov5_dir)

        # Dynamically find the most recent experiment folder (exp, exp1, exp2, etc.)
        exp_folders = [f for f in os.listdir(output_dir) if f.startswith('exp')]
        latest_exp_folder = max(exp_folders, key=lambda f: int(f[3:]) if f[3:].isdigit() else 0)
        output_path = os.path.join(output_dir, latest_exp_folder)
        labels_path = os.path.join(output_path, 'labels')

        predictions = []
        result_image_path = None

        # Loop through label files to parse predictions
        for file_name in os.listdir(labels_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(labels_path, file_name), 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()  # e.g., class_id, confidence, bbox
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            confidence = float(parts[1])
                            class_name = class_names[class_id]
                            predictions.append(class_name)  # Use list instead of set

        if not predictions:
            return jsonify({"error": "No detections found"}), 400

        # Find the detected image from the output folder
        for file_name in os.listdir(output_path):
            if file_name.endswith(('.jpg', '.png')):
                result_image_path = os.path.join(output_path, file_name)
                break

        if result_image_path is None:
            return jsonify({"error": "No result image found"}), 500

        # Ensure the result image is placed in the static folder for access
        static_image_path = os.path.join('static', 'images', os.path.basename(result_image_path))
        os.makedirs(os.path.dirname(static_image_path), exist_ok=True)  # Ensure the directory exists
        os.rename(result_image_path, static_image_path)

        # Return predictions and image path for frontend
        prediction_data = {
            "predictions": predictions,
            "image_path": static_image_path.replace("\\", "/")  # Use forward slashes for web
        }
        return jsonify(prediction_data)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

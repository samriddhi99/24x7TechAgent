
import warnings
import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import subprocess
# Suppress protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:8000", "http://localhost:8000"]}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)
    print("Saved:", input_path)

    # Ensure it's mp3/webm/etc.
    # If your FE sends .webm and you convert → put your ffmpeg code here if needed.

    # Build output paths
    output_audio = os.path.join(UPLOAD_FOLDER, "rag_output.mp3")
    output_text = os.path.join(UPLOAD_FOLDER, "rag_output.txt")

    # Prepare command
    cmd = [
        "python3", "backend/SearchEngine/audio_processingv2.1.py",
        "--input", input_path,
        "--output", output_audio,
        "--lang", "EN",
        "--mode", "hybrid",
        "--model", "gemma3:4b",
        "--save-text", output_text
    ]

    # Run command
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("Subprocess error:", result.stderr.decode())
        return jsonify({'error': 'Processing failed'}), 500

    # Read transcript from output text
    transcript = ""
    if os.path.exists(output_text):
        with open(output_text, "r") as f:
            transcript = f.read().strip()

    return jsonify({
        'message': 'File processed successfully',
        'transcript': transcript,
        'file_path': "rag_output.mp3"
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

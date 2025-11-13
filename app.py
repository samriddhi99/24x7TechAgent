
import warnings
import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess
from backend.SearchEngine.audio_processingv3 import callcenter_answer_from_bytes

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

    # Save original webm
    # Save original webm
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    print(path)

# Run ffmpeg and check for errors
    #result = subprocess.run(["ffmpeg", "-y", "-i", webm_path, "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", mp3_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    #if result.returncode != 0:
     #   print("FFmpeg error:", result.stderr.decode())
      #  return jsonify({'error': 'FFmpeg conversion failed'}), 500


    # Dummy transcript
    if path.endswith(".mp3"):
        op = callcenter_answer_from_bytes(path, "uploads")
        transcript = op["transcript"]

        return jsonify({
        'message': 'File processed successfully',
        'transcript': transcript,
        'file_path': "rag_output_answer.mp3"
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

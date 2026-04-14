
import os
import torch
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src import config
from src.dataset import IAMDataset
from src.model import CRNN
from src.utils import (
    decode_predictions, beam_search_decode_batch,
    load_checkpoint
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CHECKPOINT_PATH = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Global variables for model and mappings
model = None
char_to_idx = None
idx_to_char = None
model_meta = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    global model, char_to_idx, idx_to_char, model_meta
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        return False

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    num_classes = checkpoint.get("num_classes", len(char_to_idx) + 1)

    # Store metadata
    metrics = checkpoint.get("metrics", {})
    model_meta = {
        "epoch": checkpoint.get("epoch", "Unknown"),
        "val_loss": checkpoint.get("val_loss", 0),
        "cer": metrics.get("cer", 0),
        "word_acc": metrics.get("word_acc", 0),
        "device": str(device)
    }

    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Web] Model loaded successfully on {device}")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_info')
def model_info():
    if model is None:
        init_model()
    return jsonify(model_meta)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not init_model():
            return jsonify({'error': 'Model not initialized'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Use beam search by default for better accuracy in web app if parameters allow
        use_beam_search = request.form.get('beam_search', 'false').lower() == 'true'
        beam_width = int(request.form.get('beam_width', config.BEAM_WIDTH))

        try:
            # Create a temporary dataset item
            df = pd.DataFrame([{"image_path": filepath, "label": ""}])
            dataset = IAMDataset(df, char_to_idx, augment=False)
            
            img, _ = dataset[0]
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(img)
                if use_beam_search:
                    pred_texts = beam_search_decode_batch(preds, idx_to_char, beam_width=beam_width)
                else:
                    pred_texts = decode_predictions(preds, idx_to_char)
            
            return jsonify({
                'prediction': pred_texts[0],
                'status': 'success',
                'meta': model_meta
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    if init_model():
        app.run(debug=True, port=5000)
    else:
        print("Failed to initialize model. Please ensure best_model.pth exists.")

from flask import Flask, request, jsonify, render_template
import joblib
import traceback
import os

app = Flask(__name__)

# Model configuration
MODEL_PATH = 'spam_detector_pipeline.joblib'
LABEL_MAP = {0: "ham", 1: "spam"}

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Spam detection model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("index.html")  # Make sure index.html is in templates folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json(force=True)
        
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        text = data["text"]
        
        if not isinstance(text, str):
            return jsonify({"error": "'text' must be a string"}), 400

        prediction = model.predict([text])[0]
        
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba([text])[0][1])
        
        return jsonify({
            "prediction": LABEL_MAP[int(prediction)],
            "probability_spam": probability,
            "text_preview": text[:50] + "..." if len(text) > 50 else text
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "error",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
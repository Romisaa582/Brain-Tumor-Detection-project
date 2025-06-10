from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# تحميل النموذج المدرب
model = load_model("brain_mri_model.keras")
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# دالة التنبؤ
def predict_image(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = image.resize((299, 299))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 299, 299, 3)

    preds = model.predict(img_array)[0]
    result = {class_names[i]: float(preds[i]) for i in range(4)}
    return result

# نقطة النهاية لاستقبال الصور
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        img_bytes = file.read()
        result = predict_image(img_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# تشغيل التطبيق في Spaces
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

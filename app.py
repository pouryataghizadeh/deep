from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# =========================
# MODELLERİ YÜKLE
# =========================

# Ürün öneri modeli
crop_model = tf.keras.models.load_model("crop_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Meyve taze/çürük modeli
fruit_model = load_model("meyve_modeli.keras")

# Meyve sınıfları
class_names = [
'Taze Elma','Taze Muz','Taze Üzüm','Taze Guava',
'Taze Hünnap','Taze Portakal','Taze Nar','Taze Çilek',
'Çürük Elma','Çürük Muz','Çürük Üzüm','Çürük Guava',
'Çürük Hünnap','Çürük Portakal','Çürük Nar','Çürük Çilek'
]


# =========================
# ANA SAYFA
# =========================

@app.route('/')
def home():
    return render_template("index.html")


# =========================
# ÜRÜN ÖNERİSİ
# =========================

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

        scaled_features = scaler.transform(features)

        prediction = crop_model.predict(scaled_features)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        crop_name = label_encoder.inverse_transform([predicted_class])[0]

        return render_template(
            "index.html",
            prediction_text=f"Önerilen Ürün: {crop_name}",
            confidence_text=f"Güven Oranı: %{confidence:.2f}"
        )

    except Exception as e:
        print(e)
        return render_template(
            "index.html",
            prediction_text="Hata oluştu",
            confidence_text=""
        )


# =========================
# MEYVE ANALİZİ
# =========================

@app.route('/predict_harvest', methods=['POST'])
def predict_harvest():

    file = request.files['file']

    filepath = "temp.jpg"

    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))

    x = image.img_to_array(img)/255.0

    x = np.expand_dims(x, axis=0)

    preds = fruit_model.predict(x)

    result = class_names[np.argmax(preds)]

    confidence = float(np.max(preds))*100

    return jsonify({
        "result": result,
        "probability": f"%{confidence:.2f}"
    })


# =========================
# SERVER
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
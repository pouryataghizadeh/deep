from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import tempfile

app = Flask(__name__)

# =========================
# MODELLERİ YÜKLE
# =========================
# Dosya yollarının ana dizinde (root) olduğundan emin olun
try:
    crop_model = tf.keras.models.load_model("crop_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    fruit_model = load_model("meyve_modeli.keras")
    print("Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")

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

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
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
        print(f"Ürün Önerisi Hatası: {e}")
        return render_template(
            "index.html",
            prediction_text="Hata oluştu, lütfen değerleri kontrol edin.",
            confidence_text=""
        )

# =========================
# MEYVE ANALİZİ (GÜNCELLENDİ)
# =========================
@app.route('/predict_harvest', methods=['POST'])
def predict_harvest():
    try:
        if 'file' not in request.files:
            return jsonify({"result": "Dosya bulunamadı", "probability": "%0.00"}), 400
            
        file = request.files['file']
        
        # Güvenli geçici dosya yönetimi (Render izinleri için)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            temp_path = temp.name

        # Resmi işle
        img = image.load_img(temp_path, target_size=(224, 224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = fruit_model.predict(x)
        result = class_names[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100

        # İşlem bitince geçici dosyayı temizle
        os.remove(temp_path)

        return jsonify({
            "result": result,
            "probability": f"%{confidence:.2f}"
        })

    except Exception as e:
        print(f"Hasat Analizi Hatası: {e}")
        return jsonify({"result": "Analiz Hatası", "probability": "%0.00"}), 500

# =========================
# SERVER YAPILANDIRMASI
# =========================
if __name__ == "__main__":
    # Render'ın dinamik port atamasını dinle, yoksa 10000 kullan
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

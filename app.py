
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import tempfile
import sys

app = Flask(__name__)

# =========================
# MODELLERİ YÜKLE (GÜVENLİ MOD)
# =========================
def load_models():
    models = {}
    try:
        # Mevcut dizindeki dosyaları kontrol et
        files = os.listdir('.')
        print(f"Dizindeki dosyalar: {files}")

        models['crop'] = tf.keras.models.load_model("crop_model.h5")
        models['scaler'] = joblib.load("scaler.pkl")
        models['label_encoder'] = joblib.load("label_encoder.pkl")
        models['fruit'] = load_model("meyve_modeli.keras")
        print("Tüm modeller başarıyla yüklendi! ✅")
    except Exception as e:
        print(f"KRİTİK HATA - Modeller yüklenemedi: {e}")
        # Hata detayını Render loglarına basar
    return models

# Uygulama başlarken modelleri belleğe al
loaded_resources = load_models()

# Meyve sınıfları
class_names = [
    'Taze Elma','Taze Muz','Taze Üzüm','Taze Guava',
    'Taze Hünnap','Taze Portakal','Taze Nar','Taze Çilek',
    'Çürük Elma','Çürük Muz','Çürük Üzüm','Çürük Guava',
    'Çürük Hünnap','Çürük Portakal','Çürük Nar','Çürük Çilek'
]

# =========================
# YOLLAR (ROUTES)
# =========================

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Formdan gelen verileri al
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        features = np.array([data])
        
        # Modeller yüklü mü kontrol et
        if 'crop' not in loaded_resources:
            return render_template("index.html", prediction_text="Hata: Model dosyaları sunucuda bulunamadı.")

        scaled_features = loaded_resources['scaler'].transform(features)
        prediction = loaded_resources['crop'].predict(scaled_features)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        crop_name = loaded_resources['label_encoder'].inverse_transform([predicted_class])[0]

        return render_template(
            "index.html",
            prediction_text=f"Önerilen Ürün: {crop_name}",
            confidence_text=f"Güven Oranı: %{confidence:.2f}"
        )

    except Exception as e:
        print(f"Ürün Önerisi Hatası: {str(e)}")
        return render_template("index.html", prediction_text=f"Hata: {str(e)}")

@app.route('/predict_harvest', methods=['POST'])
def predict_harvest():
    try:
        if 'file' not in request.files:
            return jsonify({"result": "Dosya seçilmedi", "probability": "0"}), 400
            
        file = request.files['file']
        
        # Geçici dosya oluşturma (Render Yazma İzni İçin)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            temp_path = temp.name

        # Resim işleme
        img = image.load_img(temp_path, target_size=(224, 224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Tahmin
        if 'fruit' not in loaded_resources:
            return jsonify({"result": "Meyve modeli yüklenemedi", "probability": "0"}), 500

        preds = loaded_resources['fruit'].predict(x)
        result = class_names[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100

        # Temizlik
        os.remove(temp_path)

        return jsonify({
            "result": result,
            "probability": f"%{confidence:.2f}"
        })

    except Exception as e:
        print(f"Hasat Analizi Hatası: {str(e)}")
        return jsonify({"result": f"Sistem Hatası: {str(e)}", "probability": "0"}), 500

# =========================
# SERVER AYARI
# =========================
if __name__ == "__main__":
    # Render PORT çevre değişkenini kullanır, yoksa 10000 varsayılan kalır
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

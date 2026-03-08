import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Sayfa Ayarları (Tasarım)
st.set_page_config(page_title="TarımAI | Akıllı Tarım Asistanı", page_icon="🌱", layout="centered")

# CSS ile Arka Plan ve Tasarım İyileştirmeleri (HTML'indeki gibi)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2ecc71;
        text-align: center;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        width: 100%;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık
st.markdown("<h1>🌱 TarımAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Yapay Zeka Destekli Akıllı Tarım Asistanı</p>", unsafe_allow_html=True)

# =========================
# MODELLERİ YÜKLEME (Cache ile Hızlı Yükleme)
# =========================
@st.cache_resource
def load_models():
    models = {}
    try:
        models['crop'] = tf.keras.models.load_model("crop_model.h5")
        models['scaler'] = joblib.load("scaler.pkl")
        models['label_encoder'] = joblib.load("label_encoder.pkl")
        models['fruit'] = tf.keras.models.load_model("meyve_modeli.keras")
    except Exception as e:
        st.error(f"Modeller yüklenirken hata oluştu: {e}")
    return models

loaded_resources = load_models()

class_names = [
    'Taze Elma','Taze Muz','Taze Üzüm','Taze Guava',
    'Taze Hünnap','Taze Portakal','Taze Nar','Taze Çilek',
    'Çürük Elma','Çürük Muz','Çürük Üzüm','Çürük Guava',
    'Çürük Hünnap','Çürük Portakal','Çürük Nar','Çürük Çilek'
]

# =========================
# SEKMELER (TABS)
# =========================
tab1, tab2 = st.tabs(["🌱 Ürün Önerisi", "🍎 Hasat Analizi"])

# --- SEKME 1: ÜRÜN ÖNERİSİ ---
with tab1:
    st.subheader("Toprak Analizi")
    
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Azot (N)", min_value=0.0, format="%.2f")
        K = st.number_input("Potasyum (K)", min_value=0.0, format="%.2f")
        humidity = st.number_input("Nem (%)", min_value=0.0, format="%.2f")
        rainfall = st.number_input("Yağış Miktarı (mm)", min_value=0.0, format="%.2f")
    with col2:
        P = st.number_input("Fosfor (P)", min_value=0.0, format="%.2f")
        temperature = st.number_input("Sıcaklık (°C)", min_value=0.0, format="%.2f")
        ph = st.number_input("pH Değeri", min_value=0.0, format="%.2f")

    if st.button("En Uygun Ürünü Analiz Et"):
        if 'crop' in loaded_resources:
            try:
                features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                scaled_features = loaded_resources['scaler'].transform(features)
                prediction = loaded_resources['crop'].predict(scaled_features)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                crop_name = loaded_resources['label_encoder'].inverse_transform([predicted_class])[0]
                
                st.success(f"**Önerilen Ürün:** {crop_name}")
                st.info(f"**Güven Oranı:** %{confidence:.2f}")
            except Exception as e:
                st.error(f"Analiz sırasında hata: {e}")
        else:
            st.warning("Model dosyaları eksik olduğu için tahmin yapılamıyor.")

# --- SEKME 2: HASAT ANALİZİ ---
with tab2:
    st.subheader("Profesyonel Hasat Analizi")
    st.write("Desteklenen Mahsuller: Elma, Muz, Üzüm, Portakal, Çilek, Nar, Hünnap, Guava.")
    
    uploaded_file = st.file_uploader("Mahsul fotoğrafı seçin (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Resmi göster
        image = Image.open(uploaded_file)
        st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
        
        if st.button("Hasadı Analiz Et"):
            if 'fruit' in loaded_resources:
                with st.spinner('Analiz Ediliyor...'):
                    try:
                        # Resmi modele hazırla
                        img = image.resize((224, 224))
                        x = img_to_array(img) / 255.0
                        x = np.expand_dims(x, axis=0)

                        # Tahmin yap
                        preds = loaded_resources['fruit'].predict(x)
                        predicted_idx = np.argmax(preds)
                        result = class_names[predicted_idx]
                        confidence = float(np.max(preds)) * 100

                        # Sonucu göster
                        if "Taze" in result:
                            st.success(f"Sonuç: **{result}** (Güven: %{confidence:.2f})")
                        else:
                            st.error(f"Sonuç: **{result}** (Güven: %{confidence:.2f})")
                    except Exception as e:
                        st.error(f"Görsel işlenirken hata oluştu: {e}")
            else:
                st.warning("Meyve modeli eksik olduğu için analiz yapılamıyor.")

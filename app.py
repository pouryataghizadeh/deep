import streamlit as st
import numpy as np
import tensorflow as tf
import tf_keras
import joblib
from PIL import Image

# =========================
# AKILLI MODEL YÜKLEYİCİ (HİBRİT MOTOR)
# =========================
# Bu fonksiyon modele bakar. Eğer model Keras 3 ise yeni nesil motorla açar,
# eski nesil ise otomatik olarak Keras 2 (tf_keras) motoruna geçiş yapar.
def smart_load_model(path):
    try:
        # Önce Yeni Nesil Motoru Dene (Keras 3)
        return tf.keras.models.load_model(path, compile=False)
    except:
        # Hata verirse Eski Nesil Motora Geç (Keras 2)
        return tf_keras.models.load_model(path, compile=False)

# =========================
# SAYFA AYARLARI VE TASARIM
# =========================
st.set_page_config(page_title="TarımAI | Akıllı Tarım Asistanı", page_icon="🌱", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2ecc71; text-align: center; }
    .stButton>button {
        background-color: #2ecc71; color: white; border-radius: 8px;
        width: 100%; padding: 10px; font-weight: 600; letter-spacing: 1px;
    }
    .stButton>button:hover { background-color: #27ae60; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🌱 TarımAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Yapay Zeka Destekli Akıllı Tarım Asistanı</p>", unsafe_allow_html=True)

# =========================
# MODELLERİ YÜKLEME
# =========================
@st.cache_resource
def load_models():
    models = {}
    try:
        # Akıllı yükleyici ile modelleri güvenle içeri alıyoruz
        models['crop'] = smart_load_model("crop_model.h5")
        models['scaler'] = joblib.load("scaler.pkl")
        models['label_encoder'] = joblib.load("label_encoder.pkl")
        
        models['fruit'] = smart_load_model("meyve_modeli.keras")
        models['rice'] = smart_load_model("best_rice_model.h5")
        
    except Exception as e:
        st.error(f"Modeller yüklenirken hata oluştu: {e}")
    return models

loaded_resources = load_models()

# Sınıf İsimleri
fruit_class_names = [
    'Taze Elma','Taze Muz','Taze Üzüm','Taze Guava',
    'Taze Hünnap','Taze Portakal','Taze Nar','Taze Çilek',
    'Çürük Elma','Çürük Muz','Çürük Üzüm','Çürük Guava',
    'Çürük Hünnap','Çürük Portakal','Çürük Nar','Çürük Çilek'
]

rice_class_names = [
    'Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf',
    'Leaf Blast', 'Leaf scald', 'Sheath Blight'
]

# =========================
# SEKMELER (TABS)
# =========================
tab1, tab2, tab3 = st.tabs(["🌱 Ürün Önerisi", "🍎 Hasat Analizi", "🌾 Pirinç Analizi"])

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
            st.warning("Model dosyaları eksik.")

# --- SEKME 2: HASAT ANALİZİ ---
with tab2:
    st.subheader("Profesyonel Hasat Analizi")
    st.write("Desteklenen Mahsuller: Elma, Muz, Üzüm, Portakal, Çilek, Nar, Hünnap, Guava.")
    
    uploaded_fruit = st.file_uploader("Mahsul fotoğrafı seçin", type=["jpg", "png", "jpeg"], key="fruit_upload")
    
    if uploaded_fruit is not None:
        fruit_image = Image.open(uploaded_fruit)
        st.image(fruit_image, caption='Yüklenen Fotoğraf', use_container_width=True)
        
        if st.button("Hasadı Analiz Et"):
            if 'fruit' in loaded_resources:
                with st.spinner('Analiz Ediliyor...'):
                    try:
                        img = fruit_image.resize((224, 224))
                        x = np.array(img, dtype='float32') / 255.0
                        x = np.expand_dims(x, axis=0)

                        preds = loaded_resources['fruit'].predict(x)
                        predicted_idx = np.argmax(preds)
                        result = fruit_class_names[predicted_idx]
                        confidence = float(np.max(preds)) * 100

                        if "Taze" in result:
                            st.success(f"Sonuç: **{result}** (Güven: %{confidence:.2f})")
                        else:
                            st.error(f"Sonuç: **{result}** (Güven: %{confidence:.2f})")
                    except Exception as e:
                        st.error(f"Görsel işlenirken hata oluştu: {e}")
            else:
                st.warning("Meyve modeli eksik.")

# --- SEKME 3: PİRİNÇ HASTALIK ANALİZİ ---
with tab3:
    st.subheader("AI Botanik Analiz")
    st.write("Gelişmiş Derin Öğrenme modeli ile pirinç yaprağındaki hastalıkları saniyeler içinde tespit edin.")
    
    uploaded_rice = st.file_uploader("Yaprak fotoğrafı seçin", type=["jpg", "png", "jpeg"], key="rice_upload")
    
    if uploaded_rice is not None:
        rice_image = Image.open(uploaded_rice)
        st.image(rice_image, caption='Yüklenen Yaprak', use_container_width=True)
        
        if st.button("Analizi Başlat", key="rice_btn"):
            if 'rice' in loaded_resources:
                with st.spinner('Yapay Zeka İşliyor...'):
                    try:
                        if rice_image.mode != "RGB":
                            rice_image = rice_image.convert("RGB")
                        img_resized = rice_image.resize((224, 224))
                        img_array = np.array(img_resized, dtype='float32')
                        img_array = np.expand_dims(img_array, axis=0)

                        preds = loaded_resources['rice'].predict(img_array)
                        confidence = float(np.max(preds)) * 100
                        predicted_class_idx = np.argmax(preds)
                        predicted_class = rice_class_names[predicted_class_idx]

                        if predicted_class == "Healthy Rice Leaf":
                            st.success(f"✨ Sağlıklı Yaprak (Güven Skoru: %{confidence:.2f})")
                        else:
                            st.error(f"⚠️ Teşhis: **{predicted_class}** (Güven Skoru: %{confidence:.2f})")
                            
                    except Exception as e:
                        st.error(f"Görsel işlenirken hata oluştu: {e}")
            else:
                st.warning("Pirinç modeli eksik.")

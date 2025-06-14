import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Data Loading and Model Training (Bagian ini hanya dijalankan SEKALI untuk melatih dan menyimpan model)
# Idealnya, bagian ini dijalankan terpisah dan file model disimpan, lalu hanya file model yang di-deploy.
# Namun, untuk contoh lengkap dalam satu file di Colab, kita sertakan di sini.

# Nama file dataset yang akan digunakan
file_csv = "dataset_rumah_besar.csv" # Ganti jika nama file Anda berbeda

# --- BAGIAN TRAINING MODEL (Hanya dijalankan jika model belum ada) ---
# Cek apakah file model sudah ada
model_filename = 'best_rf_model.pkl'
if not os.path.exists(model_filename):
    st.warning(f"File model '{model_filename}' tidak ditemukan. Melatih model baru...")

    try:
        df_loaded = pd.read_csv(file_csv)
    except FileNotFoundError:
        st.error(f"Error: File '{file_csv}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan script ini.")
        st.stop() # Menghentikan eksekusi Streamlit jika dataset tidak ada

    # Pisahkan fitur (X) dan label (y)
    X = df_loaded.drop('layak_beli', axis=1)
    y = df_loaded['layak_beli']

    # Bagi data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Mendefinisikan fitur numerik dan kategorikal
    numeric_features = ['harga', 'luas', 'jumlah_kamar']
    categorical_features = ['lokasi']

    # Membuat preprocessor menggunakan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Membuat Pipeline
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Definisi Parameter Grid
    param_grid_rf = {
        'classifier__n_estimators': [50, 100], # Mengurangi parameter untuk mempercepat contoh
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5]
    }

    # Melakukan GridSearchCV
    grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1) # Mengurangi CV untuk mempercepat
    grid_search_rf.fit(X_train, y_train)

    # Mendapatkan model terbaik
    best_rf_model = grid_search_rf.best_estimator_

    # Simpan model terbaik ke file
    joblib.dump(best_rf_model, model_filename)
    st.success(f"Model baru telah dilatih dan disimpan sebagai '{model_filename}'")
# --- AKHIR BAGIAN TRAINING MODEL ---


# 2. Streamlit Application
st.title("Aplikasi Prediksi Kelayakan Beli Rumah")
st.write("Aplikasi ini memprediksi apakah sebuah rumah 'layak beli' berdasarkan kriteria tertentu.")

# Muat model menggunakan caching Streamlit
@st.cache_resource
def load_model(model_path):
    """Loads the trained model from a file."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"File model '{model_path}' tidak ditemukan. Harap latih model terlebih dahulu.")
        return None

pipeline_rf = load_model(model_filename)

if pipeline_rf is not None:
    st.header("Masukkan Data Rumah Baru")

    # Input dari pengguna (termasuk input harga)
    lokasi = st.selectbox("Lokasi", ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Semarang', 'Lainnya'])
    harga = st.number_input("Harga (dalam ratusan juta)", min_value=1, value=1000, format="%d") # Menggunakan format %d untuk memastikan integer
    luas = st.number_input("Luas (dalam meter persegi)", min_value=1, value=100, format="%d")
    jumlah_kamar = st.number_input("Jumlah Kamar", min_value=1, value=3, format="%d")

    # Tombol Prediksi
    if st.button("Prediksi Kelayakan Beli"):
        # Buat DataFrame dari input pengguna
        data_baru = pd.DataFrame({
            'lokasi': [lokasi],
            'harga': [harga],
            'luas': [luas],
            'jumlah_kamar': [jumlah_kamar]
        })

        # Lakukan prediksi menggunakan pipeline
        try:
            prediksi = pipeline_rf.predict(data_baru)
            prediksi_proba = pipeline_rf.predict_proba(data_baru)

            # Tampilkan hasil prediksi
            st.subheader("Hasil Prediksi")
            if prediksi[0] == 1:
                st.success(f"Rumah ini **Layak Beli**")
            else:
                st.error(f"Rumah ini **Tidak Layak Beli**")

            # Menampilkan probabilitas dengan format yang rapi
            st.write(f"Probabilitas Layak Beli: **{prediksi_proba[0][1]:.2f}**")
            st.write(f"Probabilitas Tidak Layak Beli: **{prediksi_proba[0][0]:.2f}**")

            st.write("\nCatatan: Kriteria 'layak_beli' pada model ini didasarkan pada pola di dataset pelatihan.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

else:
    st.warning("Aplikasi tidak dapat berjalan karena file model tidak dapat dimuat.")


st.markdown("""
---
*Dibuat dengan Streamlit*
""")

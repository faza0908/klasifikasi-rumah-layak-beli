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

    file_csv = "dataset_rumah_besar.csv"
    model_filename = 'best_rf_model.pkl'

    if not os.path.exists(model_filename):
        st.warning(f"File model '{model_filename}' tidak ditemukan. Melatih model baru...")
        try:
            df_loaded = pd.read_csv(file_csv)
        except FileNotFoundError:
            st.error(f"Error: File '{file_csv}' tidak ditemukan.")
            st.stop()

        X = df_loaded.drop('layak_beli', axis=1)
        y = df_loaded['layak_beli']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        numeric_features = ['harga', 'luas', 'jumlah_kamar']
        categorical_features = ['lokasi']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        pipeline_rf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        param_grid_rf = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5]
        }

        grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search_rf.fit(X_train, y_train)
        best_rf_model = grid_search_rf.best_estimator_
        joblib.dump(best_rf_model, model_filename)
        st.success(f"Model telah dilatih dan disimpan sebagai '{model_filename}'")

    st.title("Aplikasi Prediksi Kelayakan Beli Rumah")
    st.write("Aplikasi ini memprediksi apakah sebuah rumah 'layak beli'.")

    @st.cache_resource
    def load_model(model_path):
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            st.error(f"File model '{model_path}' tidak ditemukan.")
            return None

    pipeline_rf = load_model(model_filename)

    if pipeline_rf is not None:
        st.header("Masukkan Data Rumah Baru")
        lokasi = st.selectbox("Lokasi", ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Semarang', 'Lainnya'])
        harga = st.number_input("Harga (juta)", min_value=1, value=1000, format="%d")
        luas = st.number_input("Luas (m2)", min_value=1, value=100, format="%d")
        jumlah_kamar = st.number_input("Jumlah Kamar", min_value=1, value=3, format="%d")

        if st.button("Prediksi Kelayakan Beli"):
            data_baru = pd.DataFrame({
                'lokasi': [lokasi],
                'harga': [harga],
                'luas': [luas],
                'jumlah_kamar': [jumlah_kamar]
            })

            try:
                prediksi = pipeline_rf.predict(data_baru)
                prediksi_proba = pipeline_rf.predict_proba(data_baru)
                st.subheader("Hasil Prediksi")
                if prediksi[0] == 1:
                    st.success("Rumah ini **Layak Beli**")
                else:
                    st.error("Rumah ini **Tidak Layak Beli**")
                st.write(f"Probabilitas Layak Beli: **{prediksi_proba[0][1]:.2f}**")
                st.write(f"Probabilitas Tidak Layak Beli: **{prediksi_proba[0][0]:.2f}**")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
    else:
        st.warning("Aplikasi tidak dapat berjalan karena model tidak tersedia.")

    st.markdown("""---
*Dibuat dengan Streamlit*""")

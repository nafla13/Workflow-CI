import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # 1. Tentukan path data (Relative path agar bisa dibaca GitHub runner)
    # Gunakan path yang sesuai dengan struktur di repository GitHub Anda
    data_path = 'Multi-class Weather _preprocessing' 
    
    # 2. Cek apakah file ada (Penting untuk debugging CI)
    if not os.path.exists(os.path.join(data_path, 'X_train.npy')):
        print(f"Error: Folder {data_path} atau file .npy tidak ditemukan!")
        return

    # 3. Load Data
    print("Memuat data...")
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))

    # 4. Flatten data gambar (150x150x3 -> 1D)
    X_train_flat = X_train.reshape(len(X_train), -1)

    # 5. MLflow Tracking & Training
    mlflow.sklearn.autolog()
    
    # --- PERUBAHAN DISINI ---
    # Baris mlflow.set_experiment dihapus agar tidak bentrok dengan 
    # nama eksperimen yang didefinisikan di workflow CI (main.yml).

    # MLflow akan otomatis menggunakan eksperimen aktif dari GitHub Runner.
    with mlflow.start_run(run_name="GitHub_Actions_Run"):
        model = RandomForestClassifier(n_estimators=10, random_state=42) # N kecil agar cepat
        model.fit(X_train_flat, y_train)
        
        print("Training Selesai di GitHub Actions!")
        
        # Log metric manual sebagai tambahan
        accuracy = model.score(X_train_flat, y_train)
        mlflow.log_metric("final_accuracy", accuracy)

if __name__ == "__main__":
    train_model()
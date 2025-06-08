# 🎬 Movie Recommendation System

Sistem ini merupakan implementasi dua pendekatan utama dalam sistem rekomendasi film, yaitu **Content-Based Filtering** menggunakan TF-IDF dan **Collaborative Filtering** berbasis Neural Network (Matrix Factorization).

## 🔍 Deskripsi Proyek

Tujuan dari proyek ini adalah membangun sistem yang dapat merekomendasikan film kepada pengguna berdasarkan preferensi mereka, baik dari sisi konten film maupun pola interaksi pengguna.

Proyek ini terdiri dari dua pendekatan utama:
- **Content-Based Filtering**: Merekomendasikan film berdasarkan kemiripan konten (genre).
- **Collaborative Filtering**: Merekomendasikan film berdasarkan interaksi historis pengguna terhadap film (rating).

## 📁 Struktur Direktori
 ```
├───laporan.md
├───sistem_rekomendasi.ipynb
├───sistem_rekomendasi.py
└───README.md
 ```
## 🧰 Teknologi dan Library

- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn


## ✅ Cara Menjalankan

### 1. Content-Based Filtering
- Sistem akan melakukan:
  - Pembersihan data genre.
  - Transformasi genre menjadi matriks TF-IDF.
  - Penghitungan cosine similarity.
  - Rekomendasi film berdasarkan judul film input.

### 2. Collaborative Filtering
- Sistem akan melakukan:
  - Encoding ID pengguna dan film.
  - Normalisasi rating.
  - Pembangunan dan pelatihan model neural network.
  - Prediksi rating film yang belum ditonton.
  - Rekomendasi Top-N film berdasarkan skor prediksi tertinggi.

## 📊 Hasil dan Visualisasi
- Visualisasi metrik RMSE untuk memantau performa model selama pelatihan.
- Output sistem rekomendasi berdasarkan:
  - Kemiripan konten (Content-Based Filtering).
  - Prediksi rating pengguna (Collaborative Filtering).

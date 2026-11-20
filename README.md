# Analisis Dataset Ataraxia

Proyek klasifikasi emosi (8 kelas) dari kalimat bahasa Indonesia, disertai aplikasi Streamlit untuk memprediksi emosi dari teks.

Kelas: senang, sedih, marah, takut, kaget, jijik, cape, netral.

## Struktur

```
proyek_emosi/
  app.py                    Aplikasi Streamlit
  pembersihan_data.ipynb    Pembersihan data mentah menjadi data bersih
  analisis_emosi.ipynb      Eksplorasi, model, dan evaluasi
  requirements.txt          Dependensi aplikasi
  data/
    dataset_mentah_ataraxia.csv   17.530 baris (simulasi)
    dataset_clean_ataraxia.csv    16.000 baris
```

## Menjalankan

```
pip install -r requirements.txt
streamlit run app.py
```

Jalankan dari folder yang berisi `data/`. `requirements.txt` hanya memuat dependensi aplikasi; untuk menjalankan notebook, pasang juga `jupyter`, `matplotlib`, dan `seaborn`. Urutan notebook yang disarankan: `pembersihan_data.ipynb`, lalu `analisis_emosi.ipynb`.

## Data

Aplikasi dan notebook analisis hanya memakai `data/dataset_clean_ataraxia.csv`: 16.000 kalimat pendek berlabel emosi, 2.000 per kelas.

## Metode

- Normalisasi teks: huruf kecil dan spasi dirapikan.
- Pembagian data: 80% latih dan 20% uji dengan stratifikasi label (`random_state=42`).
- Fitur: TF-IDF kata (unigram dan bigram, `min_df=2`).
- Model: Logistic Regression (`C=5`).

## Hasil

Akurasi dan F1 makro pada data uji: 98,3%.

Kalimat pada dataset memiliki pola yang seragam, sehingga akurasi tinggi belum menjamin performa yang sama pada teks dengan gaya berbeda. Hal itu belum diuji.

## Catatan

`dataset_mentah_ataraxia.csv` adalah data simulasi yang dibuat dari data bersih dengan menambahkan kotoran (variasi huruf, spasi berlebih, URL, mention, duplikat, nilai kosong, dan label tidak valid). Data ini bukan data mentah hasil pengumpulan asli. Jumlah baris sebelum dan sesudah pembersihan pada `pembersihan_data.ipynb` mencerminkan simulasi tersebut.
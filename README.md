# Analisis Dataset Ataraxia

Proyek klasifikasi emosi (8 kelas) dari kalimat bahasa Indonesia, disertai aplikasi Streamlit yang menampilkan prediksi emosi dan kutipan Marcus Aurelius dengan label yang sama.

Kelas: senang, sedih, marah, takut, kaget, jijik, cape, netral.

## Struktur

```
proyek_emosi/
  app.py                    Aplikasi Streamlit
  pembersihan_data.ipynb    Pembersihan data mentah menjadi data bersih
  analisis_emosi.ipynb      Eksplorasi, model, dan evaluasi
  requirements.txt          Daftar dependensi
  data/
    dataset_mentah_ataraxia.csv         17.530 baris (simulasi)
    dataset_clean_ataraxia.csv          16.000 baris
    dataset_markus_erelius_clean.csv    1.598 baris
```

## Menjalankan

```
pip install -r requirements.txt
streamlit run app.py
```

Jalankan dari folder yang berisi `data/`. Notebook dibuka dengan `jupyter notebook`, urutan yang disarankan: `pembersihan_data.ipynb`, lalu `analisis_emosi.ipynb`.

## Data

| Berkas | Isi | Peran |
|---|---|---|
| `dataset_clean_ataraxia.csv` | Kalimat pendek berlabel emosi, 2.000 per kelas | Data latih dan uji |
| `dataset_markus_erelius_clean.csv` | Kutipan Marcus Aurelius berlabel emosi | Sumber rekomendasi kutipan |
| `dataset_mentah_ataraxia.csv` | Versi kotor dari data Ataraxia | Demonstrasi pembersihan |

Berkas Marcus Aurelius pada sebagian versi tidak berformat CSV standar (baris dibungkus tanda kutip dan diakhiri titik koma). Pembacaan dilakukan baris demi baris di `app.py` dan notebook sehingga kedua versi terbaca.

## Metode

- Pembagian data: 80% latih dan 20% uji dengan stratifikasi label.
- Fitur: TF-IDF kata (unigram dan bigram).
- Model: Logistic Regression.

## Hasil

| Pengujian | Akurasi |
|---|---|
| Data uji Ataraxia | 98,3% |
| Kutipan Marcus Aurelius | 25,3% (acak untuk 8 kelas: 12,5%) |

Akurasi tinggi pada data uji mencerminkan pola kalimat yang seragam pada dataset Ataraxia. Model tidak generalisasi ke kutipan Marcus Aurelius yang gayanya berbeda, sehingga dataset tersebut hanya dipakai sebagai sumber kutipan, bukan untuk evaluasi.

## Catatan

`dataset_mentah_ataraxia.csv` adalah data simulasi yang dibuat dari data bersih dengan menambahkan kotoran (variasi huruf, spasi berlebih, URL, mention, duplikat, nilai kosong, dan label tidak valid). Data ini bukan data mentah hasil pengumpulan asli. Jumlah baris sebelum dan sesudah pembersihan pada notebook mencerminkan simulasi tersebut.
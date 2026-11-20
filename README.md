# Dataset Emosi Ataraxia

Pembersihan dan presentasi dataset teks bahasa Indonesia berlabel emosi (8 kelas): `senang`, `sedih`, `marah`, `takut`, `jijik`, `kaget`, `cape`, `netral`.

## Isi proyek

```
proyek/
  app.py                          aplikasi Streamlit untuk presentasi data
  requirements.txt                dependensi (streamlit, pandas)
  pembersihan_data_ataraxia.ipynb notebook pembersihan data mentah
  data/
    dataset_mentah_ataraxia.csv   data mentah (17.530 baris)
    dataset_clean_ataraxia.csv    data bersih (16.000 baris)
```

## Cara menjalankan

```
pip install -r requirements.txt
streamlit run app.py
```

Untuk mengulang pembersihan, buka `pembersihan_data_ataraxia.ipynb` di Jupyter dan jalankan semua sel. Hasilnya disimpan ke `dataset_clean_hasil.csv`.

## Proses pembersihan

| Langkah | Keterangan |
|---|---|
| Penyeragaman label | huruf kecil dan buang spasi tepi, 37 variasi menjadi 8 label |
| Buang label tidak valid | kosong, `bahagia`, `unknown`, `test`, `???`, `3` (180 baris) |
| Buang teks kosong | 150 baris |
| Bersihkan teks | hapus URL, mention, karakter zero-width, tab, spasi ganda, lalu lowercase |
| Hapus duplikat | 1.200 baris |

Total 17.530 baris menjadi 16.000 baris. Hasilnya identik dengan `dataset_clean_ataraxia.csv` (teks sama, label cocok 100%).

## Kesimpulan

1. **Data mentah cukup kotor.** Sekitar 8,7% baris (1.530 dari 17.530) dibuang karena label tidak valid, teks kosong, atau duplikat. Banyak sisanya perlu dirapikan: ratusan teks berisi URL, mention, dan karakter tak terlihat, serta label dengan penulisan tidak konsisten.
2. **Hasil akhir seimbang sempurna.** Setiap kelas berisi tepat 2.000 data, sehingga tidak perlu teknik penyeimbangan kelas saat melatih model.
3. **Tidak ada duplikat teks dan tidak ada teks dengan label bertentangan** pada data bersih.
4. **Teks pendek.** Rata-rata sekitar 50 karakter (median 45, minimum 10, maksimum 253), sehingga cocok untuk model klasifikasi teks pendek.
5. **Siap dipakai untuk pemodelan**, dengan catatan berikut:
   - Kalimat cenderung berpola mirip (misalnya diakhiri "aku senang!" atau "aku marah!"), sehingga model bisa mengandalkan kata emosi eksplisit dan kurang teruji pada kalimat yang emosinya tersirat.
   - Beberapa label perlu dicek manual. Contohnya kalimat "kakak beli mobil baru. aku merasa dihina." berlabel `netral`.
   - Belum ada pembagian data latih, validasi, dan uji. Lakukan pembagian bertingkat (stratified) agar proporsi kelas tetap sama.
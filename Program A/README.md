# IF4073-PemrosesanCitraDigital-4

## Problem A - Fruit Recognition 
Program ini merupakan program untuk mengenali jenis buah menggunakan teknik-teknik **pengolahan citra digital** untuk segmentasi objek serta ekstraksi fitur dan **machine learning (SVM-ECOC)** untuk klasifikasi. 

## 🔧 Prinsip Kerja
1. **Segmentasi Buah**  
   Gambar diproses untuk memisahkan objek buah dari background menggunakan deteksi tepi dan operasi morfologi, menghasilkan mask dan crop buah.

2. **Ekstraksi Fitur**  
   Sistem mengekstraksi fitur warna (RGB/HSV), bentuk (area, eccentricity, circularity), dan tekstur (GLCM).  
   Fitur disimpan ke file `.mat` untuk mempercepat proses training dan testing.

3. **Training Model**  
   Menggunakan fitur hasil ekstraksi, program melatih model **SVM-ECOC** yang mendukung multi-class classification.  
   Model hasil pelatihan disimpan sebagai `.mat`.

4. **Klasifikasi melalui GUI**  
   Pengguna dapat:
   - Memilih gambar input  
   - Melihat hasil segmentasi (mask + crop)  
   - Menjalankan klasifikasi  

## 🧪 Cara Training Model
#### 1. **Siapkan Dataset**
   Contoh dataset yang bisa digunakan:
   - [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
   - [Fruits-262](https://www.kaggle.com/datasets/aelchimminut/fruits262)

#### 2. **Ekstraksi Features**
 ```matlab
extractAndSaveFeatures('datasetDir', 'outputFeaturesFile.mat', doSegmentation?, forceRecompute?, percentage);
```
#### 3. **Trainning Model**
 ```matlab
trainFromSavedFeatures('features_file.mat', 'outputModel.mat');
```

## 🚀 Pengunaan GUI Fruit Detection
1. Jalankan MATLAB.  
2. Buka App Designer, lalu jalankan `problem_a.mlapp`.  
3. Klik **Load Model** dan pilih model yang diinginkan.  
4. Klik **Choose** untuk memilih gambar buah.  
5. Klik **Classify** untuk menampilkan hasil pengenalan.

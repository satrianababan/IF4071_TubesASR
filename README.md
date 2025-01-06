# Tugas Besar Pemrosesan Ucapan - IF4071
>  Pembangunan Model Automatic Speech Recognition Menggunakan Kakas 

## Anggota Kelompok
<table>
    <tr>
        <td>No.</td>
        <td>Nama</td>
        <td>NIM</td>
    </tr>
    <tr>
        <td>1.</td>
        <td>Jason Rivalino</td>
        <td>13521008</td>
    </tr>
    <tr>
        <td>2.</td>
        <td>Louis Caesa Kesuma</td>
        <td>13521069</td>
    </tr>
    </tr>
    <tr>
        <td>3.</td>
        <td>Satria Octavianus Nababan</td>
        <td>13521168</td>
    </tr>
</table>

## Table of Contents
* [Deskripsi Singkat](#deskripsi-singkat)
* [Requirements](#requirements)
* [Cara Menjalankan Program](#cara-menjalankan-program)
* [Dataset References](#dataset-references)
* [Acknowledgements](#acknowledgements)

## Deskripsi Singkat
Program yang dibuat dalam pengerjaan Tugas Besar ini merupakan program untuk Pengenalan Ucapan Otomatic (Automatic Speech Recognition). Program yang dibuat dalam Tugas Besar ini menggunakan model WAV2VEC2 yang kemudian di-fine tune agar lebih sesuai dengan penggunaannya. Dataset yang digunakan untuk pelatihan model juga berupa data berbahasa Indonesia.

## Requirements
1. Visual Studio Code
2. Python (default version: 3.10.11)
3. Library install (more detail in `requirements.txt`)

## Cara Menjalankan Program
Langkah-langkah proses setup program adalah sebagai berikut:
1. Clone repository ini
2. Program dapat dijalankan dengan menjalankan setiap cell pada `experiment.ipynb`. Program tersebut akan me-load model yang telah dilatih dan mengukur akurasi untuk setiap data input pada folder `dataset_experiment`

Langkah-langkah melakukan pelatihan model dari awal:
1. Clone repository ini
2. Lakukan segmentasi pada data latih dengan menjalankan `python src/extracting_data.py`
3. Lakukan pelatihan dengan menjalankan setiap cell pada `training model.ipynb`

## Dataset References
1. ASR-IndoCSC: An Indonesian Conversational Speech Corpus (https://magichub.com/datasets/indonesian-conversational-speech-corpus/)

## Acknowledgements
- Tuhan Yang Maha Esa
- Ibu Dessi Puji Lestari sebagai Dosen Pemrosesan Ucapan IF4071

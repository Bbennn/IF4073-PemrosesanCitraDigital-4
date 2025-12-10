# Tugas Besar 3 IF4073 - Pemrosesan Citra Digital
## Program C dan D
Program pengenalan buah, license plate, dan multiple human tracking menggunakan Deep Neural Network (DNN).

#### How to run
Ikuti langkah-langkah berikut
##### 1\. Buat Virtual Environemnt

Gunakan virtual environemnt untuk mengisolasi dependecies projek

```bash
python -m venv venv
```

##### 2\. Aktifkan Virtual Environment

Jalankan command yang sesuai dengan OS:

| Operating System | Command |
| :--- | :--- |
| **Windows (Command Prompt)** | `venv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `venv\Scripts\Activate.ps1` |
| **macOS / Linux** | `source venv/bin/activate` |

##### 3\. Install Requirements
```bash
pip install -r requirement.txt --no-cache-dir
```

##### 4\. Jalankan Program
Jika belum ada model fruit recognition. Jalankan dulu proses training model untuk fruit recognition:
```bash
python train_fruit.py
```
Setelah model tersimpan, jalankan program dengan
```bash
python gui.py
```
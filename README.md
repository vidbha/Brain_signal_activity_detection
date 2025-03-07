# EEG Spectrogram Classification

## 📌 Project Overview
A deep learning-based approach to classify harmful brain activities using EEG spectrograms. This project processes EEG signals, converts them into spectrograms, and fine-tunes a pre-trained EfficientNetV2 model to make predictions. A Django-based web application is included for real-time inference.

## 🚀 Features
✅ **Data Preprocessing**: Converts raw EEG signals (.parquet files) into spectrogram images.  
✅ **Data Augmentation**: Implements MixUp and Random Cutout to improve model generalization.  
✅ **Model Training**: Fine-tunes an EfficientNetV2 model with a custom learning rate schedule and KL-Divergence loss.  
✅ **Web Application**: Deploys a Django-based web app for model inference.  

---

## 📁 Directory Structure
```
EEG-Spectrogram-Classification/
├── best_model.keras           # Trained deep learning model.
├── sample_submission.csv      # Example submission template.
├── test.csv                   # Test dataset details.
├── uploads.npy                # Processed EEG data stored in numpy format.
│
├── data/                      # Raw EEG data files in .parquet format.
│   ├── 1a5ba1f4.parquet
│   ├── 437d4faa.parquet
│   └── ... (more EEG data files)
│
├── templates/                 # Web interface files.
│   ├── input.html
│   ├── predict.html
│
├── uploads/                   # Directory for uploaded EEG files & results.
│   ├── 1000317312.parquet
│   ├── 1000493950.parquet
│   └── __results___25_0.png   # Example output image.
│
├── webapp/                     # Django web application for deployment.
│
└── hms_project/
    ├── COMParison.ipynb        # Model comparison & analysis.
    ├── db.sqlite3              # SQLite database for Django.
    ├── manage.py               # Django management script.
    │
    ├── hms_detection/          # Main Django app for EEG detection.
    │   ├── admin.py
    │   ├── apps.py
    │   ├── models.py
    │   ├── views.py
    │   └── ... (other Django files)
    │
    ├── hms_project/            # Django project settings.
        ├── settings.py
        ├── urls.py             # Routing for the application.
        ├── wsgi.py
```

---


### 2️⃣ Set Up the Django Web App
```sh
cd webapp
python manage.py migrate
python manage.py runserver
```
The application will start at `http://127.0.0.1:8000/`. Open this URL in a browser.

---

## 🔗 API Endpoints
| Endpoint     | Description  |
|-------------|-------------|
| `/`         | Renders the web dashboard. |
| `/predict`  | Accepts an EEG spectrogram file and returns the classification result. |

---

## 🔮 Future Enhancements
✅ Experiment with Vision Transformers (ViTs) for better feature extraction.  
✅ Implement real-time EEG data streaming & inference.  
✅ Enhance the web interface with better visualization tools.  
✅ Deploy the model using Docker & cloud services (AWS, GCP, or Azure).  



EEG Spectrogram Classification Project
This repository contains the code and resources for an EEG spectrogram classification project that leverages a pre-trained efficientnetv2_b2_imagenet. The project converts raw EEG signals into spectrograms, processes and augments the data, trains a deep learning model, and finally deploys a web application for inference.

Overview
The aim of this project is to classify harmful brain activities by processing EEG data into spectrogram images and training a deep learning model using transfer learning. The approach includes:

Data Preprocessing: Converting EEG signals from Parquet files into spectrogram images.
Data Augmentation: Using techniques such as MixUp and Random Cutout to improve model generalization.
Model Training: Fine-tuning a pre-trained ResNet50 model with a custom learning rate schedule and KL-Divergence loss.
Web Application: A Django-based web app to visualize and deploy model predictions.
Directory Structure
graphql
Copy
Edit
.
├── best_model.keras                # Saved model file for the best performing model.
├── sample_submission (1).csv        # Example submission file template.
├── test.csv                         # Test dataset details.
├── uploads.npy                      # Numpy file for uploads or processed data.
│
├── data                             # Contains raw EEG data files in .parquet format.
│   ├── 1a5ba1f4-ac78-4bdc-91df-26b3a3bcc636.parquet
│   ├── 437d4faa-fd4b-4d44-9405-04c2e863c82d.parquet
│   └── ... (additional .parquet files)
│
├── templates                        # HTML and image files for the web interface.
│   ├── input.html
│   └── predict.html
│
├── uploads                          # Directory for additional upload files and results.
│   ├── 1000317312.parquet
│   ├── 1000493950.parquet
│   └── __results___25_0.png         # Example output image file.
│
└── webapp                           # Django web application for model deployment.
    └── hms_project
        ├── COMParison.ipynb         # Jupyter notebook for model comparison and analysis.
        ├── db.sqlite3               # SQLite database file for Django.
        ├── manage.py                # Django management script.
        │
        ├── hms_detection            # Main Django app for EEG detection.
        │   ├── admin.py
        │   ├── apps.py
        │   ├── models.py
        │   ├── views.py
        │   └── ... (other Django files)
        │
        └── hms_project              # Django project settings.
            ├── asgi.py
            ├── settings.py          # Configuration settings for the project.
            ├── urls.py              # URL routing for the project.
            ├── wsgi.py
            └── ... (additional configuration files)
Requirements
Python 3.x
Keras, TensorFlow or JAX (as backend)
Pandas, NumPy, OpenCV
Joblib, tqdm, Matplotlib
Django (for the web application)

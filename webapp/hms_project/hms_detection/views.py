from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import os
import uuid
import csv
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
import keras_cv
import keras
from keras import ops
import tensorflow as tf
import cv2
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
TF_ENABLE_ONEDNN_OPTS=0
def inputt(request):
    return render(request, "input.html")

UPLOAD_DIR = "C:\\old\\somethig\\vs code_files\\datasetmain_copy\\uploads\\"
def generate_unique_id():
    """Generate a unique ID."""
    return str(uuid.uuid4())
class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [400, 300]  # Input image size
    epochs = 13 # Training epochs
    batch_size = 64  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v:k for k, v in label2name.items()}

def build_augmenter(dim=CFG.image_size):
    augmenters = [
        keras_cv.layers.MixUp(alpha=2.0),
        keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0),
                                     width_factor=(0.06, 0.1)), # freq-masking
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1),
                                     width_factor=(1.0, 1.0)), # time-masking
    ]
    
    def augment(img, label):
        data = {"images":img, "labels":label}
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.5:
                data = augmenter(data, training=True)
        return data["images"], data["labels"]
    
    return augment


def build_decoder(with_labels=True, target_size=CFG.image_size, dtype=32):
    def decode_signal(path, offset=None):
    # Read .npy files and process the signal
        file_bytes = tf.io.read_file(path)
        sig = tf.io.decode_raw(file_bytes, tf.float32)
        sig = sig[1024//dtype:]  # Remove header tag

        # Calculate the number of elements to pad or truncate to make it a multiple of 400
        remainder = tf.shape(sig)[0] % 400
        if remainder != 0:
        # If the length is not a multiple of 400, pad or truncate the signal to make it so
            if remainder < 200:
            # Truncate the signal
                sig = sig[:-remainder]
            else:
            # Pad the signal
                pad_size = 400 - remainder
                sig = tf.pad(sig, [[0, pad_size]])

    # Reshape the tensor into a shape containing a multiple of 400 elements
        num_rows = tf.shape(sig)[0] // 400
        sig = tf.reshape(sig, [num_rows, 400])  # Reshape into [num_rows, 400]

        # Extract labeled subsample from full spectrogram using "offset"
        if offset is not None: 
            offset = offset // 2  # Only odd values are given
            sig = sig[:, offset:offset+300]

        # Pad spectrogram to ensure the same input shape of [400, 300]
            pad_size = tf.math.maximum(0, 300 - tf.shape(sig)[1])
            sig = tf.pad(sig, [[0, 0], [0, pad_size]])

    # Log spectrogram 
        sig = tf.clip_by_value(sig, tf.math.exp(-4.0), tf.math.exp(8.0)) # avoid 0 in log
        sig = tf.math.log(sig)

    # Normalize spectrogram
        sig -= tf.math.reduce_mean(sig)
        sig /= tf.math.reduce_std(sig) + 1e-6

    # Mono channel to 3 channels to use "ImageNet" weights
        sig = tf.tile(sig[..., None], [1, 1, 3])
        return sig

    
    def decode_label(label):
        label = tf.one_hot(label, CFG.num_classes)
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [CFG.num_classes])
        return label
    
    def decode_with_labels(path, offset=None, label=None):
        sig = decode_signal(path, offset)
        label = decode_label(label)
        return (sig, label)
    
    return decode_with_labels if with_labels else decode_signal


def build_dataset(paths, offsets=None, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=False, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter()
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = (paths, offsets) if labels is None else (paths, offsets, labels)
    
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds
def predict(request):
    if request.method == 'POST':
        if 'spectrogram_file' in request.FILES:
            spectrogram_uploaded_file = request.FILES['spectrogram_file']
            spectrogram_file_path = os.path.join(UPLOAD_DIR, spectrogram_uploaded_file.name)
            # Read spectrogram file
            spec = pd.read_parquet(spectrogram_file_path)
            spectrogram_np = spec.to_numpy()
            eeg_id = generate_unique_id()
            # Save spectrogram as npy
            np.save("C:\\old\\somethig\\vs code_files\\datasetmain_copy\\uploads.npy", spectrogram_np)
            
            # Build dataset
            test_paths = ["C:\\old\\somethig\\vs code_files\\datasetmain_copy\\uploads.npy"]
            test_ds = build_dataset(test_paths, batch_size=1, repeat=False, shuffle=False, cache=False, augment=False)
            test_df = pd.read_csv("C:\\old\\somethig\\vs code_files\\datasetmain_copy\\test.csv")
            # Load model
            model = tf.keras.models.load_model('C:\\old\\somethig\\vs code_files\\datasetmain_copy\\best_model.keras')
            preds = []
            preds = model.predict(test_ds)
            pred_df = pd.DataFrame({
                'eeg_id': [eeg_id],
                'Seizure_vote': preds[0][0],
                'LPD_vote': preds[0][1],
                'GPD_vote': preds[0][2],
                'LRDA_vote': preds[0][3],
                'GRDA_vote': preds[0][4],
                'Other_vote': preds[0][5]
            })
            pred_results = {
                'eeg_id': [eeg_id],
                'Seizure_vote': preds[0][0],
                'LPD_vote': preds[0][1],
                'GPD_vote': preds[0][2],
                'LRDA_vote': preds[0][3],
                'GRDA_vote': preds[0][4],
                'Other_vote': preds[0][5]
            }
            
            # Pass the prediction results to the template
            return render(request, 'predict.html', {'pred_results': pred_results})           
    
    return HttpResponse('File Upload Failed')
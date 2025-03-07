from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import pandas as pd
import os
import keras
from tensorflow.keras.preprocessing import image  # for image loading

os.environ["KERAS_BACKEND"] = "jax"  # you can also use tensorflow or torch
import keras_cv
import numpy as np
import matplotlib.pyplot as plt
TF_ENABLE_ONEDNN_OPTS = 0



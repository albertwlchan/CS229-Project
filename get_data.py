import os
import tensorflow as tf
import pandas as pd
import numpy as np

DIM_IMG = [256,256]

def get_data():
    inputs = tf.keras.utils.image_dataset_from_directory(
    "Data/E163L02A",
    labels = None,
    color_mode='grayscale',
    batch_size = None,
    image_size=(DIM_IMG[0], DIM_IMG[1]),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

    labeled_data = pd.read_csv("Data/E163L02A/E163L02A_labels.csv",header=1)
    labeled_data = np.array(labeled_data)[:,1:]
    labels = tf.data.Dataset.from_tensor_slices(labeled_data)
    train_dataset = tf.data.Dataset.zip((inputs,labels)).batch(32)

    return train_dataset,train_dataset
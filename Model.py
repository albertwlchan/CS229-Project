from cProfile import label
import tensorflow as tf
import pandas as pd
import numpy as np


DIM_IMG = [256,256]

SIZE_BATCH = 32



def build_model():
    """
    Build a baseline acceleration prediction network.

    The network takes one input:
        img - first frame of the video

    The output is:
        (x,y,D) - predicted center location and 

    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0],1), name='img')

    conv_1 = tf.keras.layers.Conv2D(64, (5, 5),input_shape=(DIM_IMG[1], DIM_IMG[0],1),activation = "relu")(img_input)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = tf.keras.layers.Conv2D(64, (5, 5),activation = "relu")(pool_1)
    pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = tf.keras.layers.Conv2D(64, (5, 5),activation = "relu")(pool_2)
    pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_3)

    flat_out = tf.keras.layers.Flatten()(pool_3)
    # dense_1 = tf.keras.Dense(64, activation = "relu")(flat_out)
    rad_pred = tf.keras.layers.Dense(3, name='center_radius')(flat_out) #

    ########## Your code ends here ##########

    return tf.keras.Model(inputs=[img_input], outputs=[rad_pred])

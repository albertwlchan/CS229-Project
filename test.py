import tensorflow as tf
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

DIM_IMG = [1024,1024]
LEARNING_RATE = 1e-4
SIZE_BATCH = 32
NUM_EPOCHS = 50


def build_baseline_model():
    """
    Build a baseline acceleration prediction network.

    The network takes one input:
        img - first frame of the video

    The output is:
        (x,y,D) - predicted center location and 

    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 1), name='img')
    # th_input = tf.keras.Input(shape=(1,), name='th')
    ########## Your code starts here ##########
    # TODO: Replace the following with your model from build_model().
    conv_1 = tf.keras.layers.Conv2D(1, (10, 10),input_shape=(DIM_IMG[1], DIM_IMG[0],1),activation = "relu")(img_input)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

    # conv_2 = tf.keras.layers.Conv2D(5, (10, 10),activation = "relu")(pool_1)
    # pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

    # conv_3 = tf.keras.layers.Conv2D(5, (10, 10),activation = "relu")(pool_2)
    # pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_3)

    flat_out = tf.keras.layers.Flatten()(pool_1)
    rad_pred = tf.keras.layers.Dense(3, name='center_radius')(flat_out) #

    ########## Your code ends here ##########

    return tf.keras.Model(inputs=[img_input], outputs=[rad_pred])





model = build_baseline_model()

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "Data/E163L02A",
    labels = None,
    color_mode='grayscale',
    batch_size = None,
    image_size=(1024, 1024),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

labels = pd.read_csv("Data/E163L02A/E163L02A_labels.csv",header=1)
labels = np.array(labels)

# centroid = labels[:,1:3]
# diameter = labels[:,3]

labels = np.block([[np.zeros((int(labels[0,0]),3))],[labels[:,1:]]])
# data = [[tf.RaggedTensor.centroid[i]),[diameter[i]]] for i in range(diameter.shape[0])]
labels = tf.data.Dataset.from_tensor_slices(labels)



model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=tf.keras.losses.MeanSquaredError())

model.fit(train_dataset,
              epochs=NUM_EPOCHS,
              validation_data=labels,
              steps_per_epoch=20,
            )

                  
print("Yayah")
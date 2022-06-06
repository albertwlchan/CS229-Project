import tensorflow as tf
from Model import build_model
from get_data import  get_data
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
Train = False
Resolution = 0.029555664*2 # mm/pixel (after downsizing)

def main():
    path_model =  os.path.join('./src/trained_model.h5')
    path_image = "Data/E267J02Apngs"
    # Load training and test data
    print("Loading Dataset")
    dataset = get_data(path_image,load_mask=False)
    print("loaded Dataset")

    model = tf.keras.models.load_model(path_model,custom_objects={'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)})
    model.summary()

    outputs = tf.math.argmax(model.predict(dataset), axis=-1)
    images = list(dataset.as_numpy_iterator())[1]
    diameter = np.zeros(len(outputs))
    for i in range(len(outputs)):
        diameter[i] = (max(np.sum(outputs[i],axis=1))*Resolution+max(np.sum(outputs[i],axis=0))*Resolution)/2
    
    for i in range(32):
        # c = plt.Circle(outputs[i,:2], outputs[i,2],color = "red",fill = False)
        # ax = plt.imshow(images[i].astype(np.uint8)).axes
        plt.figure(3)
        plt.subplot(1,2,1)
        plt.imshow(images[i])
        plt.title("Raw Image")
        plt.subplot(1,2,2)
        plt.imshow(outputs[i])
        plt.title("Image Segmentation Mask")
        plt.xlabel("Predicted x diameter = " + np.array2string(diameter[:32][i],precision = 5))
        plt.show()

if __name__ == "__main__":
    main()
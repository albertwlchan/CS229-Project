import tensorflow as tf
from Model import build_model, loss
from get_data import get_data
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np

DIM_IMG = [256,256]
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
Train = True

def main():
    path_model =  os.path.join('./src/trained_model.h5')

    # Load training and test data
    train_data,test_data,images = get_data()


    #Builds model if doesn't exist
    if not os.path.exists(path_model) or Train == True:
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                    loss=loss)

        #Model Training
        model.fit(train_data,
                epochs=NUM_EPOCHS,
                validation_data=train_data,
                batch_size = 32
                )
        model.save(path_model)

    else:
        # Loads Save Model
        model = tf.keras.models.load_model(path_model)

    #Run Model on Test_Dataset
    outputs = model.predict(test_data)
    images = list(images.as_numpy_iterator())
    image_copy = images[:5]
    for i in range(5):
        image_copy[i] = cv2.circle(image_copy[i],outputs[i,:2].astype(int),int(outputs[i,2]),(255,0,0),1)
        plt.imshow(image_copy[i].astype(np.uint8))
        plt.show()
        
        
    

if __name__ == "__main__":
    main()
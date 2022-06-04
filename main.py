import tensorflow as tf
from Model import build_model
from get_data import  get_data
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
Train = True
Resolution = 0.029555664*2 # mm/pixel (after downsizing)

def main():

    path_model =  os.path.join('./src/trained_model.h5')

    # Load training and test data
    print("Loading Dataset")
    train_data,test_data,diameter_ground_truth,DIM_IMG = get_data()
    print("loaded Dataset")

    #Builds model if doesn't exist
    if not os.path.exists(path_model) or Train == True:

        OUTPUT_CLASSES = 2
        print("Building Model")
        model = build_model( DIM_IMG,output_channels=OUTPUT_CLASSES) 
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])


        #Model Training
        model.fit(train_data,
                epochs=NUM_EPOCHS,
                validation_data=test_data,
                batch_size = 32,
                steps_per_epoch = 20
                )
        model.save(path_model)


    else:
        # Loads Save Model
        model = tf.keras.models.load_model(path_model,custom_objects={'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)})

    #Run Model on Test_Dataset
    outputs = tf.math.argmax(model.predict(test_data), axis=-1)
    images = list(test_data.as_numpy_iterator())[0][0]
    # image_copy = images[:5]
    diameter = np.zeros(len(outputs))
    for i in range(len(outputs)):
        diameter[i] = (max(np.sum(outputs[i],axis=1))*Resolution+max(np.sum(outputs[i],axis=0))*Resolution)/2
    mse = np.square(diameter - diameter_ground_truth).mean()

    # Accuracy
    err = np.abs(np.divide(diameter-diameter_ground_truth,diameter_ground_truth))
    acc_5 = np.count_nonzero(err < 0.05)/len(diameter)
    acc_25 = np.count_nonzero(err < 0.025)/len(diameter)
    print('Mean Squared Error is {error:.4g} mm'.format(error=mse))
    print('5 accuracy is {acc5:.4g}, and 2.5 accuracy is {acc25:.4g}'.format(acc5=acc_5,acc25=acc_25))
    for i in range(32):
        # c = plt.Circle(outputs[i,:2], outputs[i,2],color = "red",fill = False)
        # ax = plt.imshow(images[i].astype(np.uint8)).axes
        plt.subplot(2,1,1)
        plt.imshow(images[i])
        plt.subplot(2,1,2)
        plt.imshow(outputs[i])

        # image_copy[i] = cv2.circle(image_copy[i],.astype(int),int(),(255,0,0),1)
        # plt.imshow(image_copy[i].astype(np.uint8))
        plt.show()



if __name__ == "__main__":

    main()
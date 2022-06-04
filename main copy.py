import tensorflow as tf
from Model import build_model, build_model2, loss
from get_data import  get_data2
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import pix2pix

DIM_IMG = [512,512]
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
Train = False
Resolution = 0.029555664*2 # mm/pixel (after downsizing)

def main():

    path_model =  os.path.join('./src/trained_model.h5')

    # Load training and test data
    print("Loading Dataset")
    train_data,test_data,diameter_ground_truth = get_data2()
    print("loaded Dataset")

    #Builds model if doesn't exist
    if not os.path.exists(path_model) or Train == True:
        # model = build_model2()
        # model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        #             loss=tf.keras.losses.BinaryCrossentropy(from_logits = True) )
        print("Building Model")
        base_model = tf.keras.applications.MobileNetV2(input_shape=[DIM_IMG[0], DIM_IMG[1], 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
            ]
        
        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
        down_stack.trainable = False

        def unet_model(output_channels:int):
            inputs = tf.keras.layers.Input(shape=[DIM_IMG[0], DIM_IMG[1], 3])

            # Downsampling through the model
            skips = down_stack(inputs)
            x = skips[-1]
            skips = reversed(skips[:-1])

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                concat = tf.keras.layers.Concatenate()
                x = concat([x, skip])

            # This is the last layer of the model
            last = tf.keras.layers.Conv2DTranspose(
                filters=output_channels, kernel_size=3, strides=2,
                padding='same')  #64x64 -> 128x128

            x = last(x)

            return tf.keras.Model(inputs=inputs, outputs=x)

        OUTPUT_CLASSES = 2

        model = unet_model(output_channels=OUTPUT_CLASSES)
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        # tf.keras.utils.plot_model(model, show_shapes=True)

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
import tensorflow as tf
from Model import build_model, build_model2, loss
from get_data import get_data, get_data2
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import pix2pix

DIM_IMG = [256,256]
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25
Train = False
Resolution = 0.029555664*4 # mm/pixel (after downsizing)

def main():


    # def create_mask(pred_mask):
    #     pred_mask = tf.math.argmax(pred_mask, axis=-1)
    #     pred_mask = pred_mask[..., tf.newaxis]
    #     return pred_mask[0]
    # def display(display_list):
    #     plt.figure(figsize=(15, 15))

    #     title = ['Input Image', 'True Mask', 'Predicted Mask']

    #     for i in range(len(display_list)):
    #         plt.subplot(1, len(display_list), i+1)
    #         plt.title(title[i])
    #         plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    #         plt.axis('off')
    #     plt.show()

    # for images, masks in train_batches.take(2):
    #     sample_image, sample_mask = images[0], masks[0]
    #     display([sample_image, sample_mask])

    # def show_predictions(dataset=None, num=1):
    #     if dataset:
    #         for image, mask in dataset.take(num):
    #         pred_mask = model.predict(image)
    #         display([image[0], mask[0], create_mask(pred_mask)])
    #     else:
    #         display([sample_image, sample_mask,
    #                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


    path_model =  os.path.join('./src/trained_model.h5')

    # Load training and test data
    train_data,test_data,images, diameter_ground_truth = get_data2()


    #Builds model if doesn't exist
    if not os.path.exists(path_model) or Train == True:
        # model = build_model2()
        # model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        #             loss=tf.keras.losses.BinaryCrossentropy(from_logits = True) )

        base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

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
            inputs = tf.keras.layers.Input(shape=[256, 256, 3])

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
                validation_data=train_data,
                batch_size = 32
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
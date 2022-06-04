
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

DIM_IMG = [512,512]

# def get_data(): ## Straight Regression from CNN to center values
#     inputs = tf.keras.utils.image_dataset_from_directory(
#     "Data/E163L02A",
#     labels = None,
#     color_mode='rgb',
#     batch_size = None,
#     image_size=(DIM_IMG[0], DIM_IMG[1]),
#     shuffle=False,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation='bilinear',
#     follow_links=False,
#     crop_to_aspect_ratio=False)
    
#     #Load Labels
#     labeled_data = pd.read_csv("Data/E163L02A/E163L02A_labels.csv",header=1)
#     labeled_data = np.array(labeled_data)[:,[1,2,4]]/2
#     size_dataset =  labeled_data.shape[0]
#     labels = tf.data.Dataset.from_tensor_slices(labeled_data)

#     #Shuffles and batches Dataset
#     train_dataset = tf.data.Dataset.zip((inputs,labels)).batch(32).shuffle(size_dataset)

#     #Train test split
#     # num_test = int(0.2*size_dataset)
#     # test_dataset = dataset.take(num_test)
#     # train_dataset = dataset.skip(num_test) \
#     #                        .shuffle(num_train, reshuffle_each_iteration=True) \
#     #                        .repeat(None)

#     return train_dataset,train_dataset,inputs


def get_data2(): ### Creates Image Masks for RCNN
    inputs = tf.keras.utils.image_dataset_from_directory(
    "Data/E163L02A",
    labels = None,
    color_mode='rgb',
    batch_size = None,
    image_size=(DIM_IMG[0], DIM_IMG[1]),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False)

    
    

    def normalize(input_image):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image
    inputs = inputs.map(normalize,num_parallel_calls=tf.data.AUTOTUNE)
    #Load Labels
    labeled_data = pd.read_csv("Data/E163L02A/E163L02A_labels.csv",header=1)
    diameter_ground_truth = np.array(labeled_data)[:,3]
    labeled_data = np.array(labeled_data)[:,[1,2,4]]/2
    size_dataset =  labeled_data.shape[0]
    num_batches = int((size_dataset + 0.5) / 32)
    num_test = num_batches // 5  # 1
    num_train = num_batches - num_test  # 14
    

    masks = []

    for element in inputs:
            im = element
            break

    print("generating masks")
    for i in range(size_dataset):
        #3 Channel Mask
        # c = cv2.circle(np.zeros((256,256,3)),labeled_data[i,:2].astype(int),int(np.ceil(labeled_data[i,2])),(255,255,255),-1)
        # c_mask = cv2.threshold(c, 128, 255, cv2.THRESH_BINARY)[1]
        # plt.imshow(c_mask)
        # masks.append(c_mask)
        # Binary
        c = cv2.circle(np.zeros((DIM_IMG[0],DIM_IMG[1],1)),labeled_data[i,:2].astype(int),int(np.ceil(labeled_data[i,2])/2),1,-1).astype(np.uint8)
        masks.append(c)
        # plt.subplot(2,1,1)
        # plt.imshow(c)
        # plt.subplot(2,1,2)
        # plt.imshow(im.numpy().astype(np.uint8))
        # print("hi")

    
    # # c = plt.Circle((59.4, 29.2), 18.2,color = "red",fill = False)
    # ax = plt.imshow(im.numpy().astype(np.uint8)).axes
    # ax.add_patch(c)
    mask_labels = tf.data.Dataset.from_tensor_slices(masks)

    #Shuffles and batches Dataset
    
    dataset = tf.data.Dataset.zip((inputs,mask_labels)).shuffle(size_dataset, reshuffle_each_iteration=False, seed=1234) \
                  .batch(32)
    test_dataset = dataset.take(num_test)
    train_dataset = dataset.skip(num_train).shuffle(num_train, reshuffle_each_iteration=True).repeat(None)
    t = tf.data.Dataset.from_tensor_slices(diameter_ground_truth)
    t = t.shuffle(1380, reshuffle_each_iteration=False, seed=1234) \
                  .batch(32)
    t = t.take(num_test)
    diameter_ground_truth = np.array(list(t.as_numpy_iterator())).flatten()

    return train_dataset,test_dataset,diameter_ground_truth
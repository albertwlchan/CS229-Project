
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

DIM_IMG = [256,256]

def get_data(): ## Straight Regression from CNN to center values
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
    
    #Load Labels
    labeled_data = pd.read_csv("Data/E163L02A/E163L02A_labels.csv",header=1)
    labeled_data = np.array(labeled_data)[:,[1,2,4]]/4
    size_dataset =  labeled_data.shape[0]
    labels = tf.data.Dataset.from_tensor_slices(labeled_data)

    #Shuffles and batches Dataset
    train_dataset = tf.data.Dataset.zip((inputs,labels)).batch(32).shuffle(size_dataset)

    #Train test split
    # num_test = int(0.2*size_dataset)
    # test_dataset = dataset.take(num_test)
    # train_dataset = dataset.skip(num_test) \
    #                        .shuffle(num_train, reshuffle_each_iteration=True) \
    #                        .repeat(None)

    return train_dataset,train_dataset,inputs


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
    
    #Load Labels
    labeled_data = pd.read_csv("Data/E163L02A/E163L02A_labels.csv",header=1)
    labeled_data = np.array(labeled_data)[:,[1,2,4]]/4
    size_dataset =  labeled_data.shape[0]

    masks = []
    for i in range(size_dataset):
        #3 Channel Mask
        # c = cv2.circle(np.zeros((256,256,3)),labeled_data[i,:2].astype(int),int(np.ceil(labeled_data[i,2])),(255,255,255),-1)
        # c_mask = cv2.threshold(c, 128, 255, cv2.THRESH_BINARY)[1]
        # plt.imshow(c_mask)
        # masks.append(c_mask)

        # Binary
        c = cv2.circle(np.zeros((256,256,1)),labeled_data[i,:2].astype(int),int(np.ceil(labeled_data[i,2])),1,-1).astype(np.uint8)
        masks.append(c)

    # for element in inputs:
    #     im = element
    #     break
    # c = plt.Circle((59.4, 29.2), 18.2,color = "red",fill = False)
    # ax = plt.imshow(im.numpy().astype(np.uint8)).axes
    # ax.add_patch(c)
    mask_labels = tf.data.Dataset.from_tensor_slices(masks)

    #Shuffles and batches Dataset
    train_dataset = tf.data.Dataset.zip((inputs,mask_labels)).batch(32).shuffle(size_dataset, seed=1234)

    #Train test split
    # num_test = int(0.2*size_dataset)
    # test_dataset = dataset.take(num_test)
    # train_dataset = dataset.skip(num_test) \
    #                        .shuffle(num_train, reshuffle_each_iteration=True) \
    #                        .repeat(None)

    return train_dataset,train_dataset,inputs
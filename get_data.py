
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

DIM_IMG = [512,512]

def get_data(im_dir,load_mask = True): ### Creates Image Masks for RCNN
    inputs = tf.keras.utils.image_dataset_from_directory(
    im_dir,
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
    num_test = num_batches // 5  
    num_train = num_batches - num_test  
    
    if load_mask == True:
        masks = []

        print("generating masks")
        for i in range(size_dataset):
    
            c = cv2.circle(np.zeros((DIM_IMG[0],DIM_IMG[1],1)),labeled_data[i,:2].astype(int),int(np.ceil(labeled_data[i,2])/2),1,-1).astype(np.uint8)
            masks.append(c)
        
        mask_labels = tf.data.Dataset.from_tensor_slices(masks)

        #Shuffles and batches Dataset
        
        dataset = tf.data.Dataset.zip((inputs,mask_labels)).shuffle(size_dataset, reshuffle_each_iteration=False, seed=1234) \
                    .batch(32)
        test_dataset = dataset.take(num_test)
        train_dataset = dataset.skip(num_test).shuffle(num_train, reshuffle_each_iteration=True).repeat(None)
        t = tf.data.Dataset.from_tensor_slices(diameter_ground_truth)
        t = t.shuffle(1380, reshuffle_each_iteration=False, seed=1234) \
                    .batch(32)
        t = t.take(num_test)
        diameter_ground_truth = np.array(list(t.as_numpy_iterator()))

        return train_dataset,test_dataset,diameter_ground_truth,DIM_IMG
    else:
        return inputs.batch(32)
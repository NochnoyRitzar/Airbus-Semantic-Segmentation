import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from constants import ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT, TRAIN_IMAGES_PATH, SEGMENTATION_MASKS_CSV_PATH
from models.inference import custom_predict


def rle_decode(mask_rle, shape=(ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    # returns empty mask if row has no data
    if not isinstance(mask_rle, str):
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        return img.reshape(shape).T

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def save_model_training_plot(model):
    # summarize history for f1-score
    plt.plot(model.history['f1-score'])
    plt.plot(model.history['val_f1-score'])
    plt.title('model f1-score (dice score)')
    plt.ylabel('f1-score')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    plt.savefig('results/train_val_history.png')


# visualize predictions on train data
def random_validation_predictions(model):
    print('Visualizing random predictions.')
    num_predictions = 10
    # read csv to get segmentation masks
    df = pd.read_csv(SEGMENTATION_MASKS_CSV_PATH)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    # pick random pictures from train folder
    images = os.listdir(TRAIN_IMAGES_PATH)
    images = random.sample(images, num_predictions)

    for img_name in images:
        # load image
        img = tf.keras.utils.load_img(os.path.join(TRAIN_IMAGES_PATH, img_name))
        ax1.imshow(img)
        ax1.set_title('Image: ' + img_name)

        predicted_mask = custom_predict(model, img_name=img_name, test_data=False)
        ax2.imshow(predicted_mask)
        ax2.set_title('Prediction Masks')

        ground_truth = create_mask(df, img_name)
        ax3.imshow(ground_truth)
        ax3.set_title('Ground Truth')

        fig.savefig(f'results/validation/{img_name}', bbox_inches='tight', pad_inches=0.1)

    print('Saved visualization to results folder.')


# create mask from RLE strings
def create_mask(df, img_name):
    # find rows in dataframe
    masks = df['EncodedPixels'].loc[df['ImageId'] == img_name].to_list()

    final_mask = np.zeros((768, 768))
    # iterate through masks array, decode each RLE string and add to final_mask
    for mask in masks:
        final_mask += rle_decode(mask)

    return final_mask

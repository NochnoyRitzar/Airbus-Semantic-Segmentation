import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT, TEST_IMAGES_PATH, IMG_WIDTH, IMG_HEIGHT
from skimage import exposure


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


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


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


# create mask from RLE strings
def create_mask(df, img_name):
    # find rows in dataframe
    masks = df['EncodedPixels'].loc[df['ImageId'] == img_name].to_list()

    final_mask = np.zeros((768, 768))
    # iterate through masks array, decode each RLE string and add to final_mask
    for mask in masks:
        final_mask += rle_decode(mask)

    return final_mask


# apply contrast stretch to image
def contrast_stretch(img):
    p1, p99 = np.percentile(img, (0.1, 99.9))
    img_rescale = exposure.rescale_intensity(img, in_range=(p1, p99))

    return img_rescale


# show 20 predicted masks from submission file
def show_submission_images(show_submission=False):
    if show_submission:
        df = pd.read_csv('data/submission.csv')
        rows = df.sample(20).values

        for row in rows:
            img_name = row[0]
            rle = row[1]

            img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, f'test_v2/{img_name}'))

            mask = rle_decode(rle, shape=(IMG_WIDTH, IMG_HEIGHT))

            # create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(img)
            ax1.set_title(f'Original image', fontsize=15)

            ax2.imshow(mask)
            ax2.set_title('Mask image', fontsize=15)

            fig.savefig(f'results/submission/{img_name}', bbox_inches='tight', pad_inches=0.1)

        print('Saved visualization of submission masks to results/submission folder.')
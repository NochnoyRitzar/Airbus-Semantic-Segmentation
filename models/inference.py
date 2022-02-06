import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from constants import IMG_WIDTH, IMG_HEIGHT, TEST_IMAGES_PATH, TRAIN_IMAGES_PATH


def custom_predict(model, img_name, test_data=True):
    # load image either from test folder or train folder
    if test_data:
        img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, 'test_v2', img_name))
    else:
        img = tf.keras.utils.load_img(os.path.join(TRAIN_IMAGES_PATH, img_name))
    # resize image to model's shape
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], preserve_aspect_ratio=True)
    # convert image to array
    img_array = tf.keras.utils.img_to_array(img)
    # expand to shape (1, IMG_WIDTH, IMG_HEIGHT, 3)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    # predict
    result = model.predict(img_array)
    result = np.squeeze(result, axis=0)

    return result


# visualize inference
def random_image_inference(model):
    print('Visualizing inference results.')

    num_predictions = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    # pick random pictures from train folder
    images = os.listdir(os.path.join(TEST_IMAGES_PATH, 'test_v2'))
    images = random.sample(images, num_predictions)

    for img_name in images:
        img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, 'test_v2', img_name))
        ax1.imshow(img)
        ax1.set_title('Image: ' + img_name)

        predicted_mask = custom_predict(model, img_name=img_name)
        ax2.imshow(predicted_mask)
        ax2.set_title('Prediction Masks')

        fig.savefig(f'results/inference/{img_name}', bbox_inches='tight', pad_inches=0.1)

    print('Saved inference visualization to results folder.')

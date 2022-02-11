import os
import random
import click
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
from constants import IMG_WIDTH, IMG_HEIGHT, TEST_IMAGES_PATH, TRAIN_IMAGES_PATH, TRAINED_MODEL_PATH, \
    SEGMENTATION_MASKS_CSV_PATH, SUBMISSION_DF_PATH
from helper_funcs import create_mask, rle_encode, show_submission_images


# download model from google drive
def download_model(download=False):
    if download:
        if os.path.exists('checkpoints/best_mobnetv2.zip'):
            os.remove('checkpoints/best_mobnetv2.zip')
        gdd.download_file_from_google_drive(file_id='1QyEmcCX3RSZ4xoO5vtDHKRwfNyz40j68',
                                            dest_path='checkpoints/best_mobnetv2.zip',
                                            unzip=True,
                                            overwrite=True)

        print('Successfully downloaded model!')


# loads pre-trained model from pre-defined folder
def load_model():
    print('Loading pre-trained model.')

    model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)

    print('Successfully loaded pre-trained model.')
    return model


# image prediction function
def custom_predict(model, img_name, test_data=True):
    # load image either from test folder or train folder
    if test_data:
        img = tf.keras.utils.load_img(os.path.join(TEST_IMAGES_PATH, 'test_v2', img_name))
    else:
        img = tf.keras.utils.load_img(os.path.join(TRAIN_IMAGES_PATH, img_name))
    # resize image to model's shape
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], preserve_aspect_ratio=True)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    # expand to shape (1, IMG_WIDTH, IMG_HEIGHT, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # predict
    result = model.predict(img_array)
    # !!! Leave prediction with more then 50% probability
    result = result > 0.5
    # reduce size to (IMG_WIDTH, IMG_HEIGHT, 1)
    result = np.squeeze(result, axis=0)

    return result


# compare prediction to ground truth
def compare_to_ground_truth(model, compare_to_gt=False):
    if compare_to_gt:
        print('Visualizing validation prediction results.')
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

        print('Saved visualization to results/validation folder.')


# visualize prediction
def visualize_random_image_inference(model, visualize_inference=False):
    if visualize_inference:
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


def predict_all_images(model, predict_all=False):
    if predict_all:
        # read csv and extract image names
        df = pd.read_csv(SUBMISSION_DF_PATH)
        images = df['ImageId'].values

        print('Started predicting test images.')
        for img_name in images:
            mask = custom_predict(model, img_name=img_name)
            mask = np.rot90(mask, k=3)
            mask = np.fliplr(mask)
            mask_rle = rle_encode(mask)
            # update row value with RLE string
            df.loc[df['ImageId'] == img_name, ['EncodedPixels']] = mask_rle
        # save to csv
        df.to_csv('data/submission.csv', index=False)
        print('Saved prediction results in data/submission.csv')


@click.command()
@click.option('--download', is_flag=True, help='To download pre-trained model.')
@click.option('--compare_to_gt', is_flag=True, help='Predict on validation images and compare to ground truth.')
@click.option('--visualize_inference', is_flag=True, help='Predict on test images and save images.')
@click.option('--predict_all', is_flag=True, help='Takes a long time! Predict all test images and save RLE information to csv file.')
@click.option('--show_submission', is_flag=True, help='Saves decoded RLE images from submission csv into results/submission folder.')
def main(download, compare_to_gt, visualize_inference, predict_all, show_submission):
    download_model(download)
    model = load_model()
    compare_to_ground_truth(model, compare_to_gt)
    visualize_random_image_inference(model, visualize_inference)
    predict_all_images(model, predict_all)
    show_submission_images(show_submission)


if __name__ == '__main__':
    main()

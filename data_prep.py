import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from constants import TRAIN_DF_PATH, TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, TEST_IMAGES_PATH, \
    BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, SEED

# load data from training_dataframe.csv
train_df = pd.read_csv(TRAIN_DF_PATH)

# we create two instances with arguments
image_gen_args = dict(rescale=1. / 255,
                      rotation_range=90,
                      horizontal_flip=True,
                      vertical_flip=True,
                      zoom_range=0.1,
                      validation_split=0.2
                      )
image_datagen = ImageDataGenerator(**image_gen_args)

mask_gen_args = dict(rescale=1. / 255,
                     rotation_range=90,
                     horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=0.1,
                     validation_split=0.2
                     )

mask_datagen = ImageDataGenerator(**mask_gen_args)


# Create training data generators
def create_train_generator():
    train_image_gen = image_datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=TRAIN_IMAGES_PATH,
                                                        x_col='ImageId',
                                                        class_mode=None,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        subset='training',
                                                        shuffle=False,
                                                        seed=SEED)

    train_mask_gen = mask_datagen.flow_from_dataframe(dataframe=train_df,
                                                      directory=TRAIN_MASKS_PATH,
                                                      x_col='ImageId',
                                                      class_mode=None,
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                      batch_size=BATCH_SIZE,
                                                      subset='training',
                                                      shuffle=False,
                                                      color_mode='grayscale',
                                                      seed=SEED)

    train_generator = zip(train_image_gen, train_mask_gen)

    return train_generator


# Create validation data generators
def create_validation_generator():
    valid_image_gen = image_datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=TRAIN_IMAGES_PATH,
                                                        x_col='ImageId',
                                                        class_mode=None,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        subset='validation',
                                                        shuffle=False,
                                                        seed=SEED)

    valid_masks_gen = mask_datagen.flow_from_dataframe(dataframe=train_df,
                                                       directory=TRAIN_MASKS_PATH,
                                                       x_col='ImageId',
                                                       class_mode=None,
                                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       subset='validation',
                                                       shuffle=False,
                                                       color_mode='grayscale',
                                                       seed=SEED)

    valid_generator = zip(valid_image_gen, valid_masks_gen)

    return valid_generator


# Create test data generators
def create_test_generator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(directory=TEST_IMAGES_PATH,
                                                      classes=['test_v2'],
                                                      batch_size=1,
                                                      seed=SEED,
                                                      shuffle=True,
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT)
                                                      )

    return test_generator

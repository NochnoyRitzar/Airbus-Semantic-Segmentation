SEED = 2022

# Path variables
SEGMENTATION_MASKS_CSV_PATH = 'data/train_ship_segmentations_v2.csv'
TRAIN_DF_PATH = 'data/training_dataframe.csv'
TRAIN_IMAGES_PATH = 'data/train_v2/'
TRAIN_MASKS_PATH = 'data/masks_v2/'
TEST_IMAGES_PATH = 'data/'  # path is correct we specify actual folder in test generator
TRAINED_MODEL_PATH = 'checkpoints/model.h5'

# Image config
ORIG_IMG_WIDTH = 768
ORIG_IMG_HEIGHT = 768
IMG_WIDTH = 256
IMG_HEIGHT = 256
N_CHANNELS = 3

# Model variables
EPOCH_COUNT = 40
BATCH_SIZE = 10
TRAIN_SAMPLES = 2400
VALID_SAMPLES = 600
ACTIVATION = "sigmoid"  # sigmoid if binary, softmax if multiclass
BACKBONE = "resnet34"  # one of segmentation models backbones
LEARNING_RATE = 0.0005

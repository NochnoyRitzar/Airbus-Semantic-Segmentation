SEED = 2022

# Path variables
SEGMENTATION_MASKS_CSV_PATH = 'data/train_ship_segmentations_v2.csv'
TRAIN_DF_PATH = 'data/training_dataframe.csv'
TRAIN_IMAGES_PATH = 'data/train_v2/'
TRAIN_MASKS_PATH = 'data/masks_v2/'
TEST_IMAGES_PATH = 'data/'  # path is correct we specify actual folder in test generator
TRAINED_MODEL_PATH = 'checkpoints/best_mobnetv2.h5'
SUBMISSION_DF_PATH = 'data/sample_submission_v2.csv'

# Image config
ORIG_IMG_WIDTH = 768
ORIG_IMG_HEIGHT = 768
IMG_WIDTH = 256
IMG_HEIGHT = 256
N_CHANNELS = 3

# Model variables
EPOCH_COUNT = 30
BATCH_SIZE = 10
TRAIN_SAMPLES = 4000
VALID_SAMPLES = 1000
ACTIVATION = "sigmoid"  # sigmoid if binary, softmax if multiclass
LEARNING_RATE = 0.0003
BACKBONE_OPTIONS = ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
                    'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'resnext50', 'resnext101', 'seresnext50',
                    'seresnext101', 'senet154', 'densenet121', 'densenet169', 'densenet201', 'inceptionv3',
                    'inceptionresnetv2', 'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2',
                    'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

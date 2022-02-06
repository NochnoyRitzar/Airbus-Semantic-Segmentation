import tensorflow as tf
import segmentation_models as sm
from segmentation_models.losses import binary_crossentropy
from segmentation_models.metrics import f1_score
from data_prep import create_train_generator, create_validation_generator
from metrics import dice_score, bce_dice_loss
from constants import BACKBONE, ACTIVATION, BATCH_SIZE, TRAIN_SAMPLES, VALID_SAMPLES, LEARNING_RATE, EPOCH_COUNT, \
    TRAINED_MODEL_PATH
from helper_funcs import save_model_training_plot

sm.set_framework('tf.keras')


# model compile function
def compile_model():
    print('Configuring model.')

    model = sm.Unet(BACKBONE, encoder_weights='imagenet', activation=ACTIVATION)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=binary_crossentropy,
                  metrics=[f1_score])
    print(f'Compiled model with {BACKBONE} backbone, {ACTIVATION} activation.')

    return model


def train_model(model, train):
    if train:
        print('Starting to train model.')
        train_gen = create_train_generator()
        valid_gen = create_validation_generator()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('checkpoints/best_model.h5',
                                               save_best_only=True,
                                               monitor='val_f1-score',
                                               mode='max')
        ]

        train_step_size = TRAIN_SAMPLES // BATCH_SIZE
        valid_step_size = VALID_SAMPLES // BATCH_SIZE
        # fit model on generator
        result = model.fit_generator(generator=train_gen,
                                     validation_data=valid_gen,
                                     validation_steps=valid_step_size,
                                     epochs=EPOCH_COUNT,
                                     steps_per_epoch=train_step_size,
                                     verbose=1,
                                     callbacks=[callbacks])
        print('Model trained.')

        save_model_training_plot(result)
        print('Saved model training and validation history into results folder.')

        return result

    else:
        return model


def load_model(compile):
    print('Loading pre-trained model.')

    model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=compile)

    return model

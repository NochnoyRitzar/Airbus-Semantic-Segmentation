import click
import tensorflow as tf
import segmentation_models as sm
from segmentation_models.losses import binary_crossentropy
from segmentation_models.metrics import f1_score
from data_prep import create_train_generator, create_validation_generator
from constants import BACKBONE_OPTIONS, ACTIVATION, BATCH_SIZE, TRAIN_SAMPLES, VALID_SAMPLES, LEARNING_RATE, EPOCH_COUNT
from helper_funcs import save_model_training_plot

sm.set_framework('tf.keras')


@click.command()
@click.option('--backbone', type=click.Choice(BACKBONE_OPTIONS, case_sensitive=True))
# model compile function
def compile_model(backbone):
    print('Compiling model.')

    model = sm.Unet(backbone, encoder_weights='imagenet', activation=ACTIVATION)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=binary_crossentropy,
                  metrics=[f1_score])
    print(f'Compiled model with {backbone} backbone, {ACTIVATION} activation.')

    return model


def train_model(model):
    print('Creating data generators. Configuring training process.')
    train_gen = create_train_generator()
    valid_gen = create_validation_generator()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('checkpoints/best_model.h5',
                                           save_best_only=True,
                                           monitor='val_f1-score',
                                           mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             mode='min',
                                             factor=0.2,
                                             patience=2,
                                             min_lr=0.000005,
                                             verbose=1)
    ]

    train_step_size = TRAIN_SAMPLES // BATCH_SIZE
    valid_step_size = VALID_SAMPLES // BATCH_SIZE

    print('Starting training.')

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
    print('Saved model history plot into results folder. Saved model can be found in checkpoints folder.')

    return result


def evaluate_model(model):
    validation_gen = create_validation_generator()

    print('Evaluating model performance.')

    # evaluate on 200 batches (2000 images)
    loss, f1 = model.evaluate_generator(validation_gen, verbose=1, steps=200)
    print(f'Loss: {loss}, f1-score: {f1 * 100}')


if __name__ == "__main__":
    model = compile_model(standalone_mode=False)
    model = train_model(model)
    evaluate_model(model)

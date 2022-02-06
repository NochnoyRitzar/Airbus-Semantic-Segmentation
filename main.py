from models.model import compile_model, train_model, load_model
from helper_funcs import random_validation_predictions
from models.inference import random_image_inference

def get_model(load_trained=True):  # change to False if you want to compile your own model model
    if load_trained:
        model = load_model(compile=False)  # set to True if you want to compile model, set to False if want to predict after loading
    else:
        model = compile_model()

    return model


if __name__ == "__main__":
    model = get_model(load_trained=True)
    model = train_model(model, train=False)  # set to True if you want to train model
    # compare prediction to ground truth
    random_validation_predictions(model)
    # predict on testing images
    random_image_inference(model)

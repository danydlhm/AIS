from src.keras_yolo import *
from keras.layers import Dense
from keras.models import Model
from os.path import join,exists
from os import makedirs
from requests import get as get_request
from src import logger

if __name__ == '__main__':
    logger.info("Start process")
    logger.info("Load definition model")
    model = make_yolov3_model()
    weight_path = join('extras','yolov3.weights')
    if not exists(weight_path):
        makedirs('extras', exist_ok=True)
        url = 'https://pjreddie.com/media/files/yolov3.weights'
        r = get_request(url)
        with open(weight_path, 'wb') as f:
            f.write(r.content)
        del r
    logger.info("Load weights")
    weight_reader = WeightReader(weight_path)
    weight_reader.load_weights(model)
    logger.info("End load weights")
    logger.info("Add new layer for project purpose")
    # for output in model.outputs:
    #     pass
    model2 = Model(model.inputs, model.outputs)
    # weight_reader.load_weights(model)
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    photo_filename = join('extras', 'zebra.jpg')
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    # make prediction
    yhat = model2.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
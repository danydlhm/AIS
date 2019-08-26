from src.keras_yolo import *
from keras.layers import Dense, Concatenate, Flatten
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
    new_outputs = []
    up_sampling = 4
    for output in model.outputs:
        if up_sampling > 1:
            up_sampled_layer = UpSampling2D(up_sampling)(output)
        else:
            up_sampled_layer = output
        new_outputs.append(up_sampled_layer)
        up_sampling = up_sampling // 2
    # merge_layer = Concatenate(axis=[2,3])(new_outputs)
    merge_layer = concatenate(new_outputs)
    output_layer = Dense(2, activation='softmax')(Flatten()(merge_layer))
    # output_layer = Dense(32)(merge_layer)
    model2 = Model(model.inputs, output_layer)
    # weight_reader.load_weights(model)
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    photo_filename = join('extras', 'zebra.jpg')
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    # make prediction
    yhat = model.predict(image)
    logger.debug([a.shape for a in yhat])
    # logger.debug(yhat)
    yhat2 = model2.predict(image)
    # summarize the shape of the list of arrays
    logger.debug([a.shape for a in yhat2])
    # logger.debug(yhat2)
    print([a.shape for a in yhat])
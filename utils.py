import numpy as np
import cv2 as cv

def preprocess_data(img):
    image = cv.resize(cv.imread(img), (224, 224), interpolation=cv.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

def prediction(model,image,class_names):
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score   


    
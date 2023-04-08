import numpy as np
import cv2 as cv

def preprocess_data(img):
    image = cv.resize(cv.imread(img), (224, 224), interpolation=cv.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

def prediction(model,image,labels):
    pred_class = model.predict(image)[0]
    pred_class_np = np.argmax(pred_class, axis=-1) 
    label = labels[pred_class_np]
    score = max(pred_class)*100
    return label, score   


    
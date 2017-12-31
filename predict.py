import sys
import glob
import os
import argparse
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from metrics import *

CLASS0 = 0
CLASS1 = 1
target_size=(227,227)

def predict(model,img,target_size):
    img = image.load_img(img, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return np.argmax(preds),preds[0,1]

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--images",help="path to images")
    a.add_argument("--model")
    args = a.parse_args()


    model = load_model(args.model)

    #image_dir = "./" + args.images + "/*jpg"
    image_dir = args.images + "/*jpg"

    test_img_paths = [img_path for img_path in glob.glob(image_dir)]

    # Making predictions
    test_ids = []
    preds = []
    truths = []
    scores = []
    for img_path in test_img_paths:

        if 'dog' in img_path:
            truths = truths + [CLASS1]
        else:
            truths = truths + [CLASS0]

        pred,score = predict(model,img_path,target_size)
        preds = preds + [pred]
        scores = scores + [score]
        test_ids = test_ids + [img_path.rpartition('/')[-1]]

    gen_metrics(truths,preds,scores)

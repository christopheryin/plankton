import sys
import glob
import os
import argparse
import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model

#cat is 0, dog is 1 -> alphabetical...

if __name__ == "__main__":

    a = argparse.ArgumentParser()
    a.add_argument("--image")
    a.add_argument("--model")
    args = a.parse_args()

    model = load_model(args.model)
    target_size = (227,227)

    img = image.load_img(args.image,target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    animal = np.argmax(preds)
    print(animal)
    print(preds)



    """
    a = argparse.ArgumentParser()
    a.add_argument("--images",help="path to images")
    args = a.parse_args()

    image_dir = "./" + args.images + "/*jpg"
    print(image_dir)

    test_img_paths = [img_path for img_path in glob.glob(image_dir)]
    print(len(test_img_paths))

    for img_path in test_img_paths:
        print(img_path)
    """
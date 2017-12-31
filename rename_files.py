import os
import argparse

def rename_images(filepath,class_name):

    for i, filename in enumerate(os.listdir(filepath)):
        os.rename(filepath + "/" + filename,filepath + "/" + class_name + str(i) + ".jpg")

if __name__ == "__main__":

    a = argparse.ArgumentParser()
    a.add_argument("--filepath", help="path to images")
    a.add_argument("--name")
    args = a.parse_args()

    rename_images(args.filepath,args.name)
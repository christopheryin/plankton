#adapted from https://github.com/DeepLearningSandbox/DeepLearningSandbox/tree/master/transfer_learning

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


IM_WIDTH, IM_HEIGHT = 299, 299
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

def get_nb_files(dir):
    if not os.path.exists(dir):
        return 0
    cnt = 0
    for r,dirs,files in os.walk(dir):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r,dr+"/*")))
    return cnt

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
      note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
      Args:
        model: keras model
      """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model



def train(args):

    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    steps_per_epoch = nb_train_samples/batch_size

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size = (IM_WIDTH,IM_HEIGHT),
        batch_size = batch_size,
        shuffle=True
    )

    val_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size = (IM_WIDTH,IM_HEIGHT),
        batch_size = batch_size,
        shuffle=True
    )

    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer(base_model,nb_classes)

    #set up to transfer learn
    setup_to_transfer_learn(model,base_model)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto'
    )

    setup_to_finetune(model)

    history_ft = mode.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto'
    )

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
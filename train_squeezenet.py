import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Convolution2D,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from squeezenet import fire_module,SqueezeNet

IM_WIDTH, IM_HEIGHT = 227, 227 #fixed size for squeezenet
NB_EPOCHS = 3
BAT_SIZE = 32

def get_nb_files(dir):
    if not os.path.exists(dir):
        return 0
    cnt = 0
    for r,dirs,files in os.walk(dir):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r,dr+"/*")))
    return cnt

def setup_to_transfer_learn(model):
    """Freeze all layers and compile the model"""
    for layer in model.layers:
        layer.trainable = False

    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = Dropout(0.5, name='drop9')(x)
    x = Convolution2D(nb_classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def setup_to_finetune(model):
    #5 layers in final output, 7 layers per fire module, finetune last 4 fire modules = 28 + 5 = 33 layers unfrozen
    #67 layers total, 0-indexed
    #layers 0-33 should be frozen, layers 34-66 trainable

    #layer 26 = finetune last 5 fire modules

    for layer in model.layers[:11]:
        layer.trainable=False
    for layer in model.layers[11:]:
        layer.trainable=True
    model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

def train(args):

    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    steps_per_epoch = nb_train_samples/batch_size
    validation_steps = nb_val_samples/batch_size

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

    base_model = SqueezeNet()
    setup_to_transfer_learn(base_model)
    model = add_new_last_layer(base_model,nb_classes)

    #sgd = SGD(lr=0.001,decay=0.0002,momentum=0.9)
    #model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history_tl = model.fit_generator(
        generator=train_generator,
        epochs=nb_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps = validation_steps,
        class_weight="auto"
    )

    setup_to_finetune(model)

    history_ft = model.fit_generator(
        generator=train_generator,
        epochs=nb_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        class_weight="auto"
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
    plt.savefig("accuracy_plot.png")
    plt.close()

    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig("loss_plot.png")

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
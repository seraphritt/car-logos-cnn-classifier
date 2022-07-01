# dataset url: https://www.kaggle.com/datasets/volkandl/car-brand-logos
# dataset url: https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset

import cv2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model


labels = ['hyundai', 'lexus', 'mazda', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen']


def train_model():
    labels = []
    dataset_path = os.listdir('dataset')
    class_labels = []
    for item in dataset_path:
        all_classes = os.listdir('dataset' + '/' + item)
        for room in all_classes:
            class_labels.append((item, str('dataset' + '/' + item) + '/' + room))
    df = pd.DataFrame(data=class_labels, columns=['Labels', 'Image'])
    images = []
    path = 'dataset/'
    img_size = 224  # tamanho da EfficientNET B0

    for i in dataset_path:
        data_path = path + str(i)
        filenames = [i for i in os.listdir(data_path)]
        for f in filenames:
            img = cv2.imread(data_path + '/' + f)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(i)
    images = np.array(images) / 255
    y = df['Labels'].values
    y_labelencoder = LabelEncoder()
    y = y_labelencoder.fit_transform(y)
    y = y.reshape(-1, 1)
    ct = ColumnTransformer([('one_ht', OneHotEncoder(), [0])], remainder='passthrough')
    Y = ct.fit_transform(y).toarray()
    images, Y = shuffle(images, Y)
    train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.2)
    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(train_x)
    num_classes = len(dataset_path)
    size = (img_size, img_size)
    inputs = layers.Input(shape=(img_size, img_size, 3))
    # Sem transfer learning == weights=None
    outputs = EfficientNetB0(include_top=True, weights=None, classes=num_classes)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    hist = model.fit_generator(datagen.flow(train_x, train_y, batch_size=32), epochs=25,
                               validation_data=(test_x, test_y))
    print(hist.history)
    model.save('car_logo_classifier.model')
    loss, accuracy = model.evaluate(test_x, test_y)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    plot_accuracy_loss(hist)


def plot_accuracy_loss(history):
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label='acc')
    plt.plot(history.history['val_accuracy'], 'ro--', label='val_acc')
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.subplot(222)
    plt.plot(history.history['loss'], 'bo--', label='loss')
    plt.plot(history.history['val_loss'], 'ro--', label='val_loss')
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig('conv_neural_network.png')


# train_model()
modelo = load_model('car_logo_classifier.model')
imagem_teste = cv2.imread('opel.jpg')
imagem_teste = cv2.cvtColor(imagem_teste, cv2.COLOR_BGR2RGB)
imagem_teste = cv2.resize(imagem_teste, (224, 224))
prediction = modelo.predict(np.array([imagem_teste]) / 255)
print(prediction)
print(labels[np.argmax(prediction)])


# Loss: 2.3509912490844727
# Accuracy: 0.5896551609039307

# 12:27, 30/06
# Loss: 1.2170417308807373
# Accuracy: 0.6379310488700867

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network
Architecture: VGG-16

@author: cynthiahabonimana
"""
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import InputLayer
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np



class ConvolutionalNeuralNetwork:
    
    #print ("Successfully imported convolutional neural network.")
    print ("VGG-16 just joined the chat")
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for validation set
    
    def __init__(self, alpha_lr):
        self.alpha_lr = alpha_lr
        

    def split (self, data, labels):
        """
        Splits the image data and corresponding labels into training 
        and validation sets.
        
        Training sets are saved in ``trainX`` and ``trainY``
        Validation sets are saved in ``validX`` and ``validY``
        """
        
        print ("Splitting data into training and validation sets...")
        (trainX, validX, trainY, validY) = train_test_split(data, labels,
         test_size=0.25, stratify=labels, random_state=42)
        print("Size and shape of trainX: {}".format(trainX.shape))
        print("Size and shape of trainY: {}".format(trainY.shape))
        print("Size and shape of validX: {}".format(validX.shape))
        print("Size and shape of validY: {}\n".format(validY.shape))
        return  trainX, validX, trainY, validY
  
    """
    DATA AUGMENTATION
    Keep this in comments if using saving intermediate output to disk
    
    # initialize the training data to add more images
    trainAug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
 
    # initialize the validation data augmentation object (which
    # we'll be adding mean subtraction to)
    valAug = ImageDataGenerator()
 
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    trainAug.mean = mean
    #valAug.mean = mean
    """
    
    
    
    
    def first_loop_model (self, data, labels):
        
        """"
        Builds a VGG-16 model that runs for one epoch. 
        The entire image dataset is fed to the model and
        the output of the VGG-16 pretrained layers is saved into 
        ``intermediate output``.
        
        
        Returns ``intermediate output``
        """"
        
        # load VGG16, ensuring the head FC layer sets are left off, while at
        # the same time adjusting the size of the input image tensor to the
        # network
        baseModel = VGG16(weights="imagenet", include_top=False,
                          input_tensor=Input(shape=(128, 128, 3)))
     
        # show a summary of the base model
        print("[INFO] summary for base model...")
        print(baseModel.summary())
    
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(5, activation="softmax")(headModel)
     
        # place the headModel on top of the base model 
        #model_first_loop is the model that will be trained
        model_first_loop = Model(inputs=baseModel.input, outputs=headModel)
     
        # loop over all layers in the base model and we freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False
    
 
        # compiling our model 
        print("[INFO] compiling model...")
        opt = Adam(lr= self.alpha_lr)
        model_first_loop.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
 
        # training the head of the network for one epoch (all other layers
        # are frozen) 
        print("[INFO] training head...")
        H_first_loop_model = model_first_loop.fit(
                	    x=data,
                    y=labels,
                    batch_size=32,
                    epochs=1,          
                    )
        last_layer_base_model = Model(inputs=model_first_loop.input,
                                 outputs=model_first_loop.layers[18].output)
        intermediate_output = last_layer_base_model.predict(data)
        print("Size and shape of the labels dataset are: {}".format(intermediate_output.shape))
        
        return intermediate_output
    
    
    def custom_model (self, trainX, validX, trainY, validY, lb):
        
        """
        Trains the model using the transformed data from pre-trained layers of VGG-16
        Diplays information on training and validation loss perfomance
        """
         
        print ("Building custom new model...")
        customModel = Sequential()
        customModel.add (AveragePooling2D(pool_size=(4, 4), input_shape= (4,4, 512)))
        customModel.add (Flatten(name="flatten"))
        customModel.add (Dropout(0.5))
        customModel.add (Dense(64, activation="relu"))
        customModel.add (Dropout(0.5))
        customModel.add (Dense(5, activation="softmax"))
                # compile our model 
        print("[INFO] compiling custom new model...")
        opt = Adam(lr= self.alpha_lr)
        customModel.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        
        print("[INFO] training custom new model...")
        H_customModel = customModel.fit(
                	    x=trainX,
                    y=trainY,
                    validation_data=(validX, validY),
                    callbacks=[es],
                    batch_size=32,
                    epochs=400,          
                    )
        
       
        print("[INFO] evaluating the network...")
        new_predictions = customModel.predict(validX, batch_size=32)
        print(classification_report(validY.argmax(axis=1),
                                   new_predictions.argmax(axis=1), target_names=lb.classes_))
        
        #N = 110
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H_customModel.history["loss"], label="train_loss")
        plt.plot(H_customModel.history["val_loss"], label="val_loss")
        plt.plot(H_customModel.history["accuracy"], label="train_acc")
        plt.plot(H_customModel.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset using Transformed Data")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        
        
        


        
        


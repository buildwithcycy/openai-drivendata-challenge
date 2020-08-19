#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-Work in progress-
Neural Network
Architecture: ResNet

@author: cynthiahabonimana
"""

from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt


class ResidualNeuralNetwork:
    
    #print ("Successfully imported ResNet50V2.")
    print ("ResNet just joined the chat")
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for validation set
    
    def __init__(self, alpha_lr):
        self.alpha_lr = alpha_lr
        
        
    def modelResNet50V2 (self, trainX, validX, trainY, validY, lb):
        """
     Builds model using a ResNet50V2 for pre-training (baseModel)
     Additional layers are added to ResNet50V2 to form the complete model. 
     
     The entire image dataset is fed to the model and
     the output of the VGG-16 pretrained layers is saved into 
     ``intermediate output``.
          
     Returns ``intermediate output`` 
     """
        baseModel = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False,
                          input_tensor=Input(shape=(128, 128, 3)))
 
        #show a summary of the base model
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
    
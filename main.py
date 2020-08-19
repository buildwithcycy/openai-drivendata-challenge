#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification using Aerial Imagery 
@author: CynthiaHabonimana
Twitter/Github: @buildwithcycy

#####################################################################
DATASET
#####################################################################
Our data consists of a set of overhead imagery taken using 
drones in seven locations across three countries with labeled 
building footprints. Our goal is to classify each of the building
footprints with the roof material type.

The Roof Material type is associated with 5 classes:

- Concrete Cement
- Healthy Metal
- Irregular Metal
- Incomplete
- Other

We apply convolutional neural networks to classify our images.

#####################################################################
PROJECT MOTIVATION: EARTHQUAKES
#####################################################################
Earthquakes are a function of mass shifting, all other structural 
elements being equal, the lighter the roof the lower the center of 
mass and, therefore, less likely it is to collapse or crumble. 
This means that roofs made out of wood tiles or asphalt are better 
than, say, brick, tile, or a heavy metal roof. Manually identifying 
the rooftop material can take several weeks. By using machine learning,
we can rapidly identify houses which have a high likelihood of collapsing.
"""


############################################################
#STEP 1: Loading Libraries
############################################################

#pip install rasterio   # For accessing geospatial raster data
#pip install geopandas  # For easily working with geospatial data 
#pip install pyproj     # For cartographic projections and coordinate transformations

import json
import os
import sys
import tensorflow as tf


############################################################
#STEP 2: Importing images from Drivendata downloaded files
############################################################

def load_geojson (path_geojson):
    """
    Loads geojson file ``geojson``.
    """
    with open(path_geojson) as geojson:
        geojson = json.load(geojson)
    print ("Geojson file successfully imported.")
    return geojson
    

############################################################
#Executing as script
############################################################
    
def main():
    use = """Use: python main.py"""
    if len(sys.argv) > 1:
        print (use)
        return
    
    print ("Ayee! Let's classify some stuff!\n")
    import pre_processing_images as preprocessor
    import convolutional_neural_network as convnet
    
    
    
    
    #Specifying the path of the main raster image from which to crop smaller rooftop images
    fpath_tiff = '/borde_rural/borde_rural_ortho-cog.tif'
    
    #Specifying the path of the Geojson file from which to grab image information (e.g. label, coordinates)
    fpath_geojson = '/borde_rural/train-borde_rural.geojson'
    
    #Specifying the learning rate alpha
    alpha_lr =0.001
    
    
    pre_processing_images = preprocessor.PreprocessingImages()
    conv_neural_network = convnet.ConvolutionalNeuralNetwork(alpha_lr)
    
    
    #loading the geoson file into ``geojson``
    geojson = load_geojson (fpath_geojson)
    
    #storing coordinates from geojson file into ``polygons``
    polygons = pre_processing_images.storing_coordinates (geojson)
    
    #storing labels from geojson file into ``labels_df``
    labels_df = pre_processing_images.storing_labels (fpath_geojson)
    
    #extracting images from the raster image and labels from geojson file 
    data,labels = pre_processing_images.mapping_image_to_label (labels_df, polygons, fpath_tiff)
    
    #transforming our labels into binary numbers 
    labels, lb = pre_processing_images.labels_to_categorical (labels)
    
    #feeding our data and labels into the frozen layers of VGG-16
    #returns the output of last layer of VGG-16, layer18=max pooling layer
    intermediate_output = conv_neural_network.first_loop_model (data, labels)
    
    
    
    #TO-DO Call SavingFRecords to save intermediate_output to disk 
    
    #Splitting our transformed data along with labels into training and validation sets
    trainX, validX, trainY, validY = conv_neural_network.split (intermediate_output, labels)
    
    #training and making predictions with our custom model
    conv_neural_network.custom_model(trainX, validX, trainY, validY, lb)
    
    
    print ("Ending Main Program...")

if __name__ == "__main__":
    main()





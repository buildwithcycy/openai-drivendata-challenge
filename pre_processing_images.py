#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processing program to extract images and labels
@author: cynthiahabonimana
"""
import json
import cv2
import os
import geopandas as gpd
import matplotlib
import rasterio
import rasterio.plot
import pyproj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time 
import sys
from pyproj import Proj
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.plot import reshape_as_image
from sklearn.preprocessing import LabelBinarizer

class PreprocessingImages:

    def storing_coordinates (self, geojson):
        """
        Stores coordinates of each image in the array ``polygons``.
        """
        polygons = []
        for feature in geojson['features']:
            polygons.append(feature['geometry'])
        print ("Yass! Coordinates successfully stored as polygons.")
        return polygons
   
    def storing_labels (self, fpath_geojson):
        """
        Stores labels of each image in the dataframe ``labels_df``.
        """
        
        labels_df = gpd.read_file (fpath_geojson)
        print("Size and shape of the labels dataset are: {}".format(labels_df.shape))
        return labels_df
    
    def exporting_cropped_images (fpath_tiff):
        """
        Defines the folders in which to export the images
        """
        src = rasterio.open(fpath_tiff, 'r')
        outfolder_irregular = '/train/irregular'
        outfolder_healthy = '/train/healthy'
        outfolder_concrete = '/train/concrete'
        outfolder_incomplete = '/train/incomplete'
        outfolder_other = '/train/other'
        outfolder = '/train/batch'
        #os.makedirs (outfolder, exist_ok = True)
        
        
    def transforming_coordinates(self, coordinates_lists, transform):
        """
        Transforms the coordinates with the provided ``affine_transform``
    
        :param coordinates_lists: list of lists of coordinates
        :type coordinates_lists: list[list[int]]
        :param transform: transformation to apply
        :type: pyproj.Proj
        """ 
        
        transformed_coordinates_lists = []
        for coordinates_list in coordinates_lists:
            transformed_coordinates_list = []
            for coordinate in coordinates_list:
                    coordinate = tuple(coordinate)
                    transformed_coordinate = list(transform(coordinate[0], coordinate[1]))
                    transformed_coordinates_list.append(transformed_coordinate)
            transformed_coordinates_lists.append(transformed_coordinates_list)
       
    
        return transformed_coordinates_lists
    
    
    
    
    def mapping_image_to_label (self, labels_df, polygons, fpath_tiff):
        """
        Maps each image to its respective label
        Images are stored in the dataframe ``data`` and labels in ``labels`

        Each label in a .geojson file will be matched to its 
        corresponding image. This is done by using the 
        coordinates provided in each .geojson file and 
        using the Rasterio library to crop the image of 
        the corresponding building.
        """ 
        
        unread_tiff = rasterio.open(fpath_tiff)

        #Projecting the coordinates to that CRS 
        proj = Proj(init='epsg:32618')
        data = []
        labels = []
        failed = []
        
        src = rasterio.open(fpath_tiff, 'r')
        outfolder = '/train/batch'
        
        print ("Hold on tight! Mapping each image to its respective label...")
        
      
        for num, row in labels_df.iterrows():
            try:
            
                
                roof_material_num = 0
                polygon0 = polygons [num]
                polygon0['coordinates'] = self.transforming_coordinates(polygon0['coordinates'], proj)
                masked_image, out_transform = rasterio.mask.mask(src,[polygon0], filled = True, crop=True, nodata = 0)
                img_image = reshape_as_image (masked_image)
                
                #Defining the name of the image file as "buildingID+roofMaterial+png" and its path 
                img_path = os.path.join (outfolder, str (row['id'])+'-'+ str (row['roof_material'])+'.png')
       
                #swapping the color channels from RGB2BGR
                img_image = cv2.cvtColor (img_image,  cv2.COLOR_RGB2BGR) #img_image is a numpy array
                
                #resizing the image dimensions to 128x128 to match ImageNet dimensions
                img_image = cv2.resize(img_image, (128, 128))
                
                #writing the image in the file
                #cv2.imwrite (img_path, img_image)
                # update the data and labels lists, respectively
                data.append(img_image) #data is a list
                labels.append(row['roof_material'])
           
            except Exception as e:
                print (e)
                failed.append (num)
        
           
        #print number of images we failed to crop and write       
        print ("Bad News First: Failed to write", len(failed), "Images.")
        print ("Good News: Successfully mapped", len (data), "Images.")
        data = np.array(data)
        labels = np.array(labels)
        #batch = data.sample(frac=0.5, replace=False, random_state=1)
        #print("Size and shape of validY: {}\n".format(batch.shape))
        return data, labels
       
        
    def labels_to_categorical (self, labels):
           
        # perform one-hot encoding on the labels
        #args = []
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        #labels = np_utils.to_categorical(labels)
        #####################
        #to_categorical` converts labels (4353x5) into a matrix with as many
        # columns as there are classes. The number of rows
        # stays the same.
        return labels, lb 

    print ("Successfully imported preprocessing tools.")
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional but fix documentation and encapsulation
@author: cynthiahabonimana
"""
import tensorflow as tf
import numpy as np

import glob

from PIL import Image

class SavingTFRecords:
    

    def _int64_feature(value):
        """
        Converts the numeric values into features
        _int64 is used for numeric values
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    
    def _bytes_feature(value):
        """
        Converts the string/char values into features 
        bytes is used for string/char values
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    # Specifying the name of the tfRecord file
    tfrecord_filename = 'dummydataset_testing.tfrecords'

    # Initiating the writer and creating the tfrecords file.
    writer = tf.io.TFRecordWriter(tfrecord_filename)

    # Loading the location of all files - aerial image dataset
    # Considering our image dataset has apple or orange
    # The images are named as buildingnumber.png
    images = glob.glob('borde_rural/images-colombia-borde-rural/*.png')
    for image in images[:1]:
        img = Image.open(image)
        img = np.array(img.resize((128,128)))
    label = 0 if 'a1c' in image else 1
    feature = { 'label': _int64_feature(label),
              'image': _bytes_feature(img.tostring())}
    
    # Creating an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Writing the serialized example
    writer.write(example.SerializeToString())

    writer.close()
    print ("TFRecord saved!")
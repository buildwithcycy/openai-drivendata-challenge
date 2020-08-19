#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T0-DO: Fix readingTFrecords class

@author: cynthiahabonimana
"""

import tensorflow as tf 
import glob



class readingTFRecords:
 
    print ("Class readingTFRecords isn't functional yet. On your T0-DO List")
    """
    raw_image_dataset = tf.data.TFRecordDataset('something_test.tfrecords')
    
    feature_set = { 'image': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)
                   }
           
    features = tf.io.parse_single_example(raw_image_dataset, features= feature_set )
    label = features['label']
    """
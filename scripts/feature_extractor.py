import json
import traceback

import cv2

__author__ = 'ananya'

import sys

sys.path.append("..")
from scipy.misc import imresize
import numpy as np
import os
from datetime import datetime
from keras.models import load_model
from scripts.utils import triplet_loss, accuracy, l2Norm, euclidean_distance
from keras.models import Model
import io
from keras import backend as K
import mysql.connector

class FeatureExtractor(object):
    def __init__(self, path_to_model_file, embedding_layer='visnet_model', height=224, width=224):
        self.path_to_model_file = path_to_model_file
        self.embedding_layer = embedding_layer
        self.height = height
        self.width = width
        fashion_lens_model = load_model(self.path_to_model_file, custom_objects= {'triplet_loss':triplet_loss,'accuracy':accuracy, 'l2Norm':l2Norm, 'euclidean_distance':euclidean_distance})
        print("Model loaded")
        self.visnet_model = fashion_lens_model.get_layer(embedding_layer)
        self.visnet_model._make_predict_function()
        # visnet_model = Model(inputs=fashion_lens_model.input,
        #                                  outputs=fashion_lens_model.get_layer(layer_name).output)

        # self.visnet_model.summary()

        # self.mydb = mysql.connector.connect(
        #   host="localhost",
        #   user="root",
        #   passwd="root",
        #   database="fashion_lens"
        # )


    def extract_one(self, path):
        # read colored image
        # in_memory_file = io.BytesIO()
        # query_image.save(in_memory_file)
        # print(query_image)
        resized_img = None
        try:
            img = self.getImageFromPath('static/image/product/' + path)
            resized_img = imresize(img, (self.height, self.width), 'bilinear')
        except Exception as e:
            print("Exception for image", path)
            traceback.print_exc()

        embedding = self.visnet_model.predict([[resized_img]])
        
        return embedding[0]

    def extract_batch(self, img_paths, index):
        batch_size = len(img_paths)

        print(img_paths)
        # mycursor = self.mydb.cursor()

        # for i in range(len(img_paths)):
        #     sql = "INSERT INTO crop_image_vector (id, path) VALUES (%s, %s)"
        #     val = (i+1, img_paths[i])
        #     mycursor.execute(sql, val)

        # self.mydb.commit()
        fv_dict = {}
        start_time = datetime.now()
        resized_imgs = []
        for path in img_paths:
            try:
                img = self.getImageFromPath(path)
                # print(img)
                resized_imgs.append(imresize(img, (self.height, self.width), 'bilinear'))
            except Exception as e:
                print("Exception for image", path)
                traceback.print_exc()

        embedding = self.visnet_model.predict([resized_imgs])

        return embedding

    def getImageFromPath(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)

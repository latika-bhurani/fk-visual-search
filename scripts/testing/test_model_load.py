import base64
# import h
import numpy as np
import io
from PIL import Image
import keras
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
# from keras import backend as K
from flask import jsonify
from flask import Flask
from flask import request
import tensorflow as tf
import keras.backend as K
from keras.layers import concatenate, Flatten, Input, Lambda
from keras.applications.vgg16 import preprocess_input
# from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
from keras.models import load_model
# from PIL import Image
# import numpy as np
import flask
# import io
import h5py 


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

print('Loading Model')
my_model = load_model('fashion_lens_triplet_model.h5', custom_objects={'l2Norm': l2Norm, 'triplet_loss':triplet_loss, 'accuracy':accuracy, 'euclidean_distance':euclidean_distance})
print(" * Model Loaded! ")

query_img_path = '.\\structured_images\\wtbi_skirts_query_crop_256\\3.jpg'

query_img = image.load_img(query_img_path, target_size=(224, 224))
query_img_data = image.img_to_array(query_img)
query_img_data = np.expand_dims(query_img_data, axis=0)
query_img_data = preprocess_input(query_img_data)

print(query_img_data.shape)
dummy_pos = np.random.randn(*query_img_data.shape)
dummy_neg = np.random.randn(*query_img_data.shape) 

# b = np.random.randn(*a.shape)
# print(b.shape)
# print(my_model.predict([query_img_data, dummy_pos, dummy_neg]))

# [<tf.Tensor 'query_input:0' shape=(?, 224, 224, 3) dtype=float32>, <tf.Tensor 'positive_input:0' shape=(?, 224, 224, 3) dtype=float32>, <tf.Tensor 'negative_input:0' shape=(?, 224, 224, 3) dtype=float32>]
print(my_model.get_input_at(0)[0])





# vizNet_model.get_input_at(0)
## extract the feature vector


model_name = 'vizNet_model'
layer_name = 'viznet_embedding'
# extract_feature_model = Model(inputs=my_model.get_input_at(0)[0], 
#                                 outputs=my_model.get_layer(model_name).get_layer(layer_name).output)
extract_feature_model = Model(inputs=my_model.get_layer(model_name).get_input_at(0), 
                                outputs=my_model.get_layer(model_name).get_layer(layer_name).output)


# feature_vector = my_model.predict([query_img_data, dummy_pos, dummy_neg])
feature_vector = extract_feature_model.predict(query_img_data)
print(feature_vector)
print(feature_vector.shape)
print(feature_vector.tolist())


# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# query_input (InputLayer)        (None, 224, 224, 3)  0                                            
# __________________________________________________________________________________________________
# positive_input (InputLayer)     (None, 224, 224, 3)  0                                            
# __________________________________________________________________________________________________
# negative_input (InputLayer)     (None, 224, 224, 3)  0                                            
# __________________________________________________________________________________________________
# vizNet_model (Model)            (None, 4096)         164846038   query_input[0][0]                
#                                                                  positive_input[0][0]             
#                                                                  negative_input[0][0]             
# __________________________________________________________________________________________________
# pos_dist (Lambda)               (None, 1)            0           vizNet_model[1][0]               
#                                                                  vizNet_model[2][0]               
# __________________________________________________________________________________________________
# neg_dist (Lambda)               (None, 1)            0           vizNet_model[1][0]               
#                                                                  vizNet_model[3][0]               
# __________________________________________________________________________________________________
# stacked_dists (Lambda)          (None, 2, 1)         0           pos_dist[0][0]                   
#                                                                  neg_dist[0][0]                   
# ===============================================================================================
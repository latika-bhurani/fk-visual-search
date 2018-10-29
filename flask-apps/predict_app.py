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
from flaskext.mysql import MySQL



app = Flask(__name__)

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

def get_model():
	global model
	model = load_model('fashion_lens_triplet_model.h5', custom_objects={'l2Norm': l2Norm, 'triplet_loss':triplet_loss, 'accuracy':accuracy, 'euclidean_distance':euclidean_distance})
	print(" * Model Loaded! ")

def preprocess_image(image, target):
	if image.mode != 'RGB':
		image = image.convert("RGB")
	# resize image input and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)

	# return the processed image
	return image

print(" * Loading Keras Mode...")
get_model()

# making custom model on top to extract feature vector

model_name = 'vizNet_model'
layer_name = 'viznet_embedding'

extract_feature_model = Model(inputs=model.get_layer(model_name).get_input_at(0), 
                                outputs=model.get_layer(model_name).get_layer(layer_name).output)

# feature_vector = extract_feature_model.predict(query_img_data)

## db connection to access and store features

# this will take out the top k images // or will run knn on the whole db for that matter
# will return a list of ids that we can pass back to the client side
# the client side will contain the images aand it can chose to display them from there
def get_db_connection(app):
	mysql = MySQL()
	# mySql Configurations
	app.config['MYSQL_DATABASE_USER'] = 'root'
	app.config['MYSQL_DATABASE_PASSWORD'] = ''
	app.config['MYSQL_DATABASE_DB'] = 'fashion_lens'
	app.config['MYSQL_DATABASE_HOST'] = 'localhost'

	mysql.init_app(app)
	cur = mysql.get_db().cursor()

	return cur

def get_top_k(cur, image_id):
	sql = 'SELECT features FROM image_features WHERE image_num = {}'.format(image_id)
	cur.execute(sql)
	rv = cur.fetchall()
	return str(rv)



@app.route('/predict', methods=['POST', 'GET'])
def predict():

	message = request.get_json(force=True)
	encoded = message["image"]
	image_id = message["imageId"]
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	print('image id is ', image_id)

	# preprocess the image and prepare it for classification
	processed_image = preprocess_image(image, target=(224,224))

	# classify the input image and then initialize the list
	# of predictions to return to the client
	print('about to predict')
	preds = extract_feature_model.predict(processed_image).tolist()
	# results = ima
	print(len(preds[0]))


	## db connection
	cur = get_db_connection(app)
	similar = get_top_k(cur, image_id)

	print(similar)

	response = {
		'prediction':(similar)
	}

	return jsonify(response)

	# cur.get_db
if __name__ == '__main__':
	app.run(debug=True)
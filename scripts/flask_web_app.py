# run_keras_server.py
# Reference --> https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

## Keras REST API

'''
load_model: Used to load our trained Keras model and prepare it for inference.

prepare_image: This function preprocesses an input image prior to passing it through 
				our network for prediction. If you are not working with image data you may want to consider changing the name to a more generic prepare_datapoint and applying any scaling/normalization you may need.

predict: The actual endpoint of our API that will classify
		 the incoming data from the request and return the results to the client.
'''

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None



# link to the fashion_lens_model
def load_model():
	### load model##
	'''
	this will do the followinig:

	1) instantiate architecture
	2) load the weights

	'''
	global model
	model
	pass

def prepare_image(image, target):

	'''
	<<FUNCTIONALITIES>>
	Accepts an input image
	Converts the mode to RGB (if necessary)
	Resizes it to 224x224 pixels (the input spatial dimensions for ResNet)
	Preprocesses the array via mean subtraction and scaling
	'''
	# image - incoming image
	# target - target size of the image 

	# change the image mode to RGB if not already
	if image.mode != 'RGB':
		image = image.convert('RGB')

	# resize image input and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image




@app.route("/predict", methods=["POST"]):

def predict():
	## initialize data directory that will be returned from the view
	data = {"success": False}

	# ensure an image was properly uploaded to out endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224,224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			# results = ima
			## !! decode predictions into the list !!
			data["predictions"] = []

			# loop over the results and add them to the list of predictions
			for (// data returned from prediction) in results[0]:
				res = {"label": , "probability": , }
				data["predictions"].append(res)

			# indicate that the result was successful
			data["success"] = True

	return flask.jsonify(data)


'''
data dictionary is returned to the client

it contains the boolean "success" 
it also conatains predictions we have made

To accept the incoming data we check if:

The request method is POST (enabling us to send arbitrary data to the endpoint, including 
images, JSON, encoded-data, etc.)
An image has been passed into the files attribute during the POST
We then take the incoming data and:

Read it in PIL format
Preprocess it
Pass it through our network
Loop over the results and add them individually to the data["predictions"] list
Return the response to the client in JSON format

If you're working with non-image data you should remove the request.files code and either
parse the raw input data yourself or utilize request.get_json() to automatically parse the
input data to a Python dictionary/object. Additionally, consider giving following tutorial 
a read which discusses the fundamentals of Flask's request object.
''' 

## launching our service

# if this is the main thread of execution, first load the model and then
# start the server

if __name__ == '__main__':
	print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

	load_model()
	app.run()



# https://www.youtube.com/watch?v=TW_ck9NDMGI
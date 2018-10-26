from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np


def get_features(img_path):
	model = VGG16(weights='imagenet', include_top=False)
	# model.summary()

	# img_path = '.\\structured_images\\wtbi_dresses_query_crop_256\\'+image_number+'.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)

	vgg16_feature = model.predict(img_data)

	vgg16_feature_np = np.array(vgg16_feature)
	vgg16_feature_flatten = vgg16_feature_np.flatten()

	return vgg16_feature_flatten
	# print(vgg16_feature.shape)
	# print(vgg16_feature)

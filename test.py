from annoy import AnnoyIndex
import random
from scripts.utils import euclidean_distance
from scripts.feature_extractor import FeatureExtractor
import numpy as np

# '6289', 'data/structured_images/skirts_256\\19829.jpg'
# '6290', 'data/structured_images/skirts_256\\19830.jpg'
# '6291', 'data/structured_images/skirts_256\\19831.jpg'
# '15073', 'data/structured_images/skirts_256\\44715.jpg'
# '15370', 'data/structured_images/skirts_256\\61184.jpg'
# '15935', 'data/structured_images/skirts_256\\76011.jpg'
# '15939', 'data/structured_images/skirts_256\\76200.jpg'

search_index = AnnoyIndex(4096, 'euclidean')
search_index.load('scripts/data/indices/skirts_ann_index.ann')   

model_path = 'scripts/data/model/fashion_lens_model.h5'
feature_extractor = FeatureExtractor(model_path)

vector1 = feature_extractor.extract_one('data/structured_images/skirts_256\\19829.jpg')
vector2 = feature_extractor.extract_one('data/structured_images/skirts_256\\19830.jpg')
vector3 = feature_extractor.extract_one('data/structured_images/skirts_256\\19831.jpg')


print(vector1)
print(vector2)
print(vector3)
print("Euclidean distance ::::::::::: ")
print(np.linalg.norm(vector1-vector2))

print("distance 6289-6290 : ", np.linalg.norm(vector1-vector2))
print("distance 6289-6291 : ", np.linalg.norm(vector1-vector3))
print("distance 6290-6291 : ", np.linalg.norm(vector2-vector3))

print("ANN Distance :::::::::::::::")

print("Get nearest neighbours")
# nearest_neightbours = search_index.get_nns_by_vector(query_image_vector, 20, include_distances = True)
print("Get nearest neighbours")
print("6289-6289 : ", search_index.get_distance(6289, 6289));
print("6289-6290 : ", search_index.get_distance(6289, 6290));
print("6289-6291 : ", search_index.get_distance(6289, 6291));
print("6290-6291 : ", search_index.get_distance(6290, 6291));
print("6289-15073 : ", search_index.get_distance(6289, 15073));
print("6289-15370 : ", search_index.get_distance(6289, 15370));
print("6289-15935 : ", search_index.get_distance(6289, 15935));
print("6289-15939 : ", search_index.get_distance(6289, 15939));

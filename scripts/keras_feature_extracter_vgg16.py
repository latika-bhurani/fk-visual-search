# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# import numpy as np
# import csv

# count = 0
# features_list = [[]]
# with open('triplets_skirts_10_sample_new1.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter=',')
# 	for row in csv_reader:
# 		print(row)
# 		features_list.append(row)
# 		model = VGG16(weights='imagenet', include_top=False)
# 		# model.summary()

# 		img_path = row[0]
# 		# print(img_path)
# 		# exit(0)
# 		img = image.load_img(img_path, target_size=(224, 224))
# 		img_data = image.img_to_array(img)
# 		img_data = np.expand_dims(img_data, axis=0)
# 		img_data = preprocess_input(img_data)

# 		vgg16_feature = model.predict(img_data)
# 		features_list[count]

# 		print(vgg16_feature.shape)
# 		# print(vgg16_feature)





# 		count+=1
# 		if count == 10:
# 			exit(0)




# # model = VGG16(weights='imagenet', include_top=False)
# # model.summary()

# # img_path = '.\\structured_images\\wtbi_dresses_query_crop_256\\1.jpg'
# # img = image.load_img(img_path, target_size=(224, 224))
# # img_data = image.img_to_array(img)
# # img_data = np.expand_dims(img_data, axis=0)
# # img_data = preprocess_input(img_data)

# # vgg16_feature = model.predict(img_data)

# # print(vgg16_feature.shape)
# # print(vgg16_feature)
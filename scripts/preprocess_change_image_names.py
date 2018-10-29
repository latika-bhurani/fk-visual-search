# change the names of the images from 9chat long to int sized in images folder
# example --> 000001111.jpeg --> 1111.jpg

import os
import sys

path = os.getcwd()+'..\\images-dataset'
print(path)

counter = 0
for filename in os.listdir(path):
	if len(filename) > 12:
		new_filename = str(int(filename[:9])) + '.jpg'
		os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
	if len(filename) < 12:
		new_filename = str(filename.split('.')[0])+'.jpg'
		os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

print('done')

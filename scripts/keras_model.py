from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16

from scripts.normalizor import LRN
import tensorflow as tf
from keras.layers import Concatenate, Flatten, Input, merge
from keras.applications.vgg16 import preprocess_input
import numpy as np
import csv


class VisNet:

    ALPHA = 0.2

    # Build VisNet
    def build_base_network(self, X):
        # input = Input(shape=X.T.shape, name='image_input')

        # VGG Layer
        vgg = VGG16(weights='imagenet', include_top=False)

        # model = VGG16(weights='imagenet', include_top=False)
        # model.summary()

        # img_path = '.\\structured_images\\wtbi_dresses_query_crop_256\\'+image_number+'.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)

        # can be removed
        # vgg.add(Dense(4096))

        # Shallow Layers
        shallow_1 = Sequential()
        shallow_1.add(Conv2D(3, kernel_size=8, strides=8, activation='relu', name='subsample_1'))
        # shallow_1.add(MaxPooling2D(pool_size=4, strides=8, name='subsample_1'))
        shallow_1.add(Conv2D(96, kernel_size=8, strides=4, activation='relu', name='conv1'))
        # shallow_1.add(LRN(name='conv1_norm'))
        shallow_1.add(BatchNormalization(axis=-1, momentum=0.99,epsilon=0.001))                                                       # ??
        shallow_1.add(MaxPooling2D(pool_size=7, strides=4, border_mode='valid', name='pool1'))
        shallow_1.add(Flatten())

        shallow_2 = Sequential()
        shallow_2.add(Conv2D(3, kernel_size=8, strides=8, activation='relu', name='subsample_1'))
        # shallow_2.add(MaxPooling2D(pool_size=8, strides=8, name='subsample_2'))
        shallow_2.add(Conv2D(96, kernel_size=8, strides=4, activation='relu', name='conv2'))
        # shallow_2.add(LRN(name='conv2_norm'))                                                       # ??
        shallow_2.add(BatchNormalization(axis=-1, momentum=0.99,epsilon=0.001))
        shallow_2.add(MaxPooling2D(pool_size=3, strides=2, name='pool2'))
        shallow_2.add(Flatten())

        # concatenated shallow layer
        shallow = Concatenate([shallow_1, shallow_2])

        # TODO
        # combine VGG16  and shallow layer output
        shallow.add(Dense(4096))


        ## we can use batch normaliaztion from keras with axis specifies as across channels 
        shallow.add(LRN(name='shallow_norm', alpha=8191, n=8191, beta=0.5))

        return model

    def build_model(self):

        ## read images from the triplets
        for image_path in query_image:
            img = image.load_img(image_path, target_size=224,224)
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)







        query_input = Input((224, 224, 3))
        positive_input = Input((224, 224, 3))
        negative_input = Input((224, 224, 3))

        query_embedding = self.build_base_network(query_input)
        positive_embedding = self.build_base_network(positive_input)
        negative_embedding = self.build_base_network(negative_input)

        loss = merge([query_embedding, positive_embedding, negative_embedding],
                     mode=triplet_loss, output_shape=(1,))
        model = Model(inputs=[query_input, positive_input, negative_input],
                      outputs=loss)

    def triplet_loss(self, X):

        query, positive, negative = X

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(query, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(query, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        return loss


if __name__ == '__main__':
    
    triplet_csv_path = ''
    batch_size = 20
    # read images from triplets
    query_image = []
    negative_image = []
    positive_image = []
    with open(triplet_csv_path) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        # for every 20 rows/inputs

        query_image.append(row[0])
        positive_image.append(row[1])
        negative_image.append(row[2])
        
        if row_count == 20:
            build_model(query_image, positive_image, negative_image)
            query_image = []
            positive_image = []
            negative_image = []

          # print(vgg16_feature)




'''

layer {
  name: "norm1_q2"
  type: "LRN"
  bottom: "conv1_q2"
  top: "norm1_q2"
  lrn_param {
    local_size: 5
    alpha: 9.99999974738e-05
    beta: 0.75
  }
'''
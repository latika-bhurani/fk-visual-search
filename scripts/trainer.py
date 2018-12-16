from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
import keras.backend as K
from keras.layers import concatenate, Flatten, Input, Lambda
from keras.applications.vgg16 import preprocess_input
import numpy as np
import csv
import itertools
from utils import l2Norm, triplet_loss, euclidean_distance
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
from keras.callbacks import ModelCheckpoint
import keras
import os

class Trainer:

    def __init__(self):

        self._alpha = 0.2
        self.model_path = "../model/fashion_lens_model.h5"
        self.triplet_dir = "../data/"

    def build_base_network(self):

        # input
        input = Input(shape=(224,224,3), name='image_input')

        # VGG Layer for high level features
        vgg = VGG16(weights='imagenet', include_top=False)

        # to make vgg layers non-trainable
        for layer in vgg.layers:
            layer.trainable = False

        vgg = vgg(input)

        # Add the fully connected layers on top
        vgg = Flatten(name='flatten')(vgg)
        vgg = Dense(4096, activation='relu', name='vgg16_fc1')(vgg)
        vgg = Dense(4096, activation='relu', name='vgg16_fc2')(vgg)
        vgg = Lambda(l2Norm, output_shape=[4096])(vgg)

        # Shallow layers for fine grained attributes of the image

        # shallow layer 1
        shallow1 = Conv2D(3, kernel_size=4, strides=4, activation='relu', name='subsample_1')(input)
        shallow1 = Conv2D(96, kernel_size=8, strides=4, activation='relu', name='conv1')(shallow1)
        shallow1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(shallow1)  # ??
        shallow1 = MaxPooling2D(pool_size=7, strides=4, padding='valid', name='pool1')(shallow1)
        shallow1 = Flatten()(shallow1)
        shallow1 = Dense(1536)(shallow1)

        # shallow layer 2
        shallow2 = Conv2D(3, kernel_size=8, strides=8, activation='relu', name='subsample_2')(input)
        shallow2 = Conv2D(96, kernel_size=8, strides=4, activation='relu', name='conv2')(shallow2)
        shallow2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(shallow2)
        shallow2 = MaxPooling2D(pool_size=3, strides=2, name='pool2')(shallow2)
        shallow2 = Flatten()(shallow2)
        shallow2 = Dense(1536)(shallow2)

        # concatenated shallow layers
        shallow = concatenate([shallow1, shallow2])
        shallow = Lambda(l2Norm, output_shape=[3072])(shallow)

        # combine shallow and vgg16 models
        merged = concatenate([vgg, shallow])
        merged = Dense(4096)(merged)
        merged = Lambda(l2Norm, output_shape=[4096], name='visnet_embedding')(merged)

        # visnet model
        return Model(input, merged, name='visnet_model')


    """
    Build main model to train data the data on using triplet loss to minimize
    Model takes the triplets of query, positive and negative image of size (224, 224, 3)
    as input
    """
    def build_model(self):

        query_input = Input((224,224,3), name='query_input')
        positive_input = Input((224,224,3), name='positive_input')
        negative_input = Input((224,224,3), name='negative_input')

        # common embedding network to share weights
        visnet = self.build_base_network()

        # fetch embeddings for triplet
        query_embedding = visnet(query_input)
        positive_embedding = visnet(positive_input)
        negative_embedding = visnet(negative_input)

        # use euclidean distance for both query-positive and query-negative embedding
        positive_dist = Lambda(euclidean_distance, name='pos_dist')([query_embedding, positive_embedding])
        negative_dist = Lambda(euclidean_distance, name='neg_dist')([query_embedding, negative_embedding])

        # stack the two embeddings
        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists'
        )([positive_dist, negative_dist])

        # create final model
        fashion_lens_model = Model([query_input, positive_input, negative_input], stacked_dists,
                                   name='fashion_lens_triplet_model')
        fashion_lens_model.compile(optimizer="adam", loss=triplet_loss)

        return fashion_lens_model


    '''
    Method to train the model on training data batch by batch
    '''
    def train(self, model, triplet_csv, batch_size = 64):

        tbCallBack = keras.callbacks.TensorBoard(log_dir="../logs/{}".format(time()), histogram_freq=0, write_graph=True,
                                                 write_images=True)
        total_train_size = 83852

        n_batches = int(total_train_size / batch_size)

        triplet_csv = os.path.join(self.triplet_dir + triplet_csv)

        print("Start training")
        print("Number of batches : ", n_batches)
        # testing purpose -- run only 1 batch
        for batch_num in range(n_batches):
            print("Batch Num : ", batch_num)
            triplet_batch = trainer.create_batch(triplet_csv, batch_num, batch_size)

            query_image, positive_image, negative_image = triplet_batch
            batch_len = len(query_image)
            Y_train = np.random.randint(2, size=(1, 2, batch_len)).T

            # save final model
            model.fit([query_image, positive_image, negative_image], Y_train, epochs=30, validation_split=0.2, callbacks=[tbCallBack])

        # save final model
        model.save(self.model_path)

    # TODO
    # tune batch size
    def create_batch(self, triplet_csv_path, batch_number, batch_size):

        query_image = np.zeros((batch_size,224,224,3))
        negative_image = np.zeros((batch_size,224,224,3))
        positive_image = np.zeros((batch_size,224,224,3))

        with open(triplet_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            count = 0
            for row in itertools.islice(csv_reader, batch_number*batch_size, (batch_number+1)*batch_size):

                if (os.path.exists(row[0]) and os.path.exists(row[1]) and os.path.exists(row[2])) :
                    query_img_path = row[0]
                    query_img = image.load_img(query_img_path, target_size=(224, 224))
                    query_img_data = image.img_to_array(query_img)
                    query_img_data = np.expand_dims(query_img_data, axis=0)
                    query_img_data = preprocess_input(query_img_data)
                    query_image[count] = query_img_data

                    positive_img_path = row[1]
                    positive_img = image.load_img(positive_img_path, target_size=(224, 224))
                    positive_img_data = image.img_to_array(positive_img)
                    positive_img_data = np.expand_dims(positive_img_data, axis=0)
                    positive_img_data = preprocess_input(positive_img_data)
                    positive_image[count] = positive_img_data


                    negative_img_path = row[2]
                    negative_img = image.load_img(negative_img_path, target_size=(224, 224))
                    negative_img_data = image.img_to_array(negative_img)
                    negative_img_data = np.expand_dims(negative_img_data, axis=0)
                    negative_img_data = preprocess_input(negative_img_data)
                    negative_image[count] = negative_img_data

                    count += 1

        return query_image, positive_image, negative_image


    '''
    Visualization for the specified layer

    TODO
        - need to update as per requirement
    '''
    def layer_to_visualize(self, layer, model, image_to_visualize):
        inputs = [K.learning_phase()] + model.inputs

        _convout1_f = K.function(inputs, [layer.output])

        def convout1_f(X):
            # The [0] is to disable the training phase flag
            return _convout1_f([0] + [X])

        convolutions = convout1_f(image_to_visualize)
        convolutions = np.squeeze(convolutions)

        print('Shape of conv:', convolutions.shape)

        n = convolutions.shape[0]
        n = int(np.ceil(np.sqrt(n)))

        # Visualization of each filter of the layer
        fig = plt.figure(figsize=(12, 8))
        for i in range(len(convolutions)):
            ax = fig.add_subplot(n, n, i + 1)
            ax.imshow(convolutions[i], cmap='gray')


trainer = Trainer()
trainer.train(trainer.build_model(), 'triplets_skirts_10_sample_new.csv')

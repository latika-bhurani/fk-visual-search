import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.utils import plot_model
# import t.keras.backend as K



# from scripts.normalizor import LRN
#import tensorflow as tf
import keras.backend as K
from keras.layers import concatenate, Flatten, Input, Lambda
from keras.applications.vgg16 import preprocess_input
import numpy as np
import csv
import itertools




ALPHA = 0.2

# Build VisNet
def build_base_network(triplet_batch):
    # input = Input(shape=(224,224,3), name='image_input')


    # '''
    ## THIS IS TO BE DONE IF ALL THE FULLY CONNECTED LAYERS ARE TO BE REPLACED

    # VGG Layer
    vgg16_model = VGG16(weights='imagenet', include_top=False)        #input_shape=(224,224,3)

    ###########################################
    # for layer in resnet_model.layers:
    # layer.trainable = False
    ############################################


    # own input image
    input = Input(shape=(224,224,3), name='image_input')

    # use the generated model
    output_vgg16 = vgg16_model(input)

    # Add the fully connected layers on top

    vgg16_flatten = Flatten(name='flatten')(output_vgg16)
    vgg16_fc1 = Dense(4096, activation='relu', name='vgg16_fc1')(vgg16_flatten)
    vgg16_fc2 = Dense(4096, activation='relu', name='vgg16_fc2')(vgg16_fc1)
    # x = Dense(8, activation='softmax', name='predictions')(x)
    # '''

    print(vgg16_fc2.shape) # (?, 4096)
    print(vgg16_fc2) # Tensor("vgg16_fc2/Relu:0", shape=(?, 4096), dtype=float32)

    # net = Lambda(l2Norm, output_shape=[512])(net)

    vgg16_l2 = Lambda(l2Norm, output_shape=[4096])(vgg16_fc2)

    print('vgg16_l2')
    print(vgg16_l2.shape) # (?, 4096)
    print(vgg16_l2)

    '''
    ONLY THE LAT LAYER IS TO BE UPDATED

    # Generate a model with all layers (with top)
    vgg16 = VGG16(weights=None, include_top=True)

    #Add a layer where input is the output of the  second last layer 
    x = Dense(8, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    #Then create the corresponding model 
    my_model = Model(input=vgg16.input, output=x)
    my_model.summary()
    '''
    
    shallow_1 = Conv2D(3, kernel_size=4, strides=4, activation='relu', name='subsample_1')(input)
    # shallow_1.add(MaxPooling2D(pool_size=4, strides=8, name='subsample_1'))
    s1_conv1 = Conv2D(96, kernel_size=8, strides=4, activation='relu', name='conv1')(shallow_1)
    # shallow_1.add(LRN(name='conv1_norm'))
    s1_bn1 = BatchNormalization(axis=-1, momentum=0.99,epsilon=0.001)(s1_conv1)                                                      # ??
    s1_maxp1 = MaxPooling2D(pool_size=7, strides=4, padding='valid', name='pool1')(s1_bn1)
    print('s1 max pooled shape', s1_maxp1)
    s1_flat1 = Flatten()(s1_maxp1)
    print('s1 flatten shape', s1_flat1.shape)
    shallow1_dense = Dense(1536)(s1_flat1)

    #print('shallow_1 ', end='\\n\\n')
    # print(shallow_1.summary())
    print(shallow1_dense)
    print(shallow1_dense.shape)
    # exit(0)


    
    shallow_2 = Conv2D(3, kernel_size=8, strides=8, activation='relu', name='subsample_2')(input)
    # shallow_2.add(MaxPooling2D(pool_size=8, strides=8, name='subsample_2'))
    s2_conv1 = Conv2D(96, kernel_size=8, strides=4, activation='relu', name='conv2')(shallow_2)
    # shallow_2.add(LRN(name='conv2_norm'))                                                       # ??
    s2_bn1 = BatchNormalization(axis=-1, momentum=0.99,epsilon=0.001)(s2_conv1)
    s2_maxp1 = MaxPooling2D(pool_size=3, strides=2, name='pool2')(s2_bn1)
    print('s2 max pooled shape', s2_maxp1)
    s2_flat1 = Flatten()(s2_maxp1)
    print('s2 flatten shape', s2_flat1.shape)
    shallow2_dense = Dense(1536)(s2_flat1)


    #print('shallow_2 ', end='\n\n')

    print(shallow2_dense)
    print(shallow2_dense.shape)

    # concatenated shallow layer
    shallow = concatenate([shallow1_dense, shallow2_dense])

    shallow_l2 = Lambda(l2Norm, output_shape=[3072])(shallow)

    # combine cshallow and vgg16 models
    vgg16_shallow_combined = concatenate([vgg16_l2, shallow_l2])

    vgg16_shallow_combined_dense = Dense(4096)(vgg16_shallow_combined)
    vgg16_shallow_combined_l2 = Lambda(l2Norm, output_shape=[4096],name='viznet_embedding')(vgg16_shallow_combined_dense)

    vizNet_model = Model(input, vgg16_shallow_combined_l2, name='vizNet_model')  ##########!!!!!!!!!!!!!!!!!!!!!!! input ?????

    input_shape=(224,224,3)
    query_input = Input(shape=(224,224,3), name='query_input')
    positive_input = Input((224, 224, 3), name='positive_input')
    negative_input = Input((224, 224, 3),name='negative_input')

    query_embedding = vizNet_model(query_input)
    positive_embedding = vizNet_model(positive_input)
    negative_embedding = vizNet_model(negative_input)    


    positive_dist = Lambda(euclidean_distance, name='pos_dist')([query_embedding, positive_embedding])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([query_embedding, negative_embedding])

    stacked_dists = Lambda( 
                lambda vects: K.stack(vects, axis=1),
                name='stacked_dists'
    )([positive_dist, negative_dist])

    fashion_lens_model = Model([query_input, positive_input, negative_input], stacked_dists, name='fashion_lens_triplet_model')

    fashion_lens_model.compile(optimizer="adam", loss=triplet_loss, metrics=[accuracy])

    query_image, positive_image, negative_image = triplet_batch
    batch_len = len(query_image)
    # print(batch_len)
    Y_train = np.random.randint(2, size=(1,2,batch_len)).T

    print(query_image.shape)
    # exit(0)

    # save final model
    fashion_lens_model.fit([query_image, positive_image, negative_image], Y_train, epochs=50, validation_split=0.2)
    fashion_lens_model.save('./model/fashion_lens_triplet_model.h5')

    #
    #
    # #####################################
    # # Plot model
    #
    # import pydot
    #
    # plot_model(vizNet_model, to_file='viznet_model.png')
    #
    # plot_model(fashion_lens_model, to_file='fashion_lens_model.png')


    ######################################


    ## Extracting features from last layer and saving to database
    model_name = 'vizNet_model'
    layer_name = 'viznet_embedding'
#    extract_feature_model = Model(inputs=fashion_lens_model.input, 
#                                    outputs=fashion_lens_model.get_layer(model_name).get_layer(layer_name).output)
#    feature_vector = extract_feature_model.predict(query_image)
    # feat_vec = features.predict(query_image)

#    print(feature_vector)

#	sme = 'vizNet_model'
#    layer_name = 'viznet_embedding'
    # layer_name = 'viznet_embedding/l2_normalize:0'
    # extract_feature_model = Model(inputs=fashion_lens_model.input, 
    #                                 outputs=fashion_lens_model.get_layer(model_name).get_layer(layer_name).output)
    extract_feature_model = Model(inputs=vizNet_model.get_input_at(0), 
                                    outputs=vizNet_model.get_layer(layer_name).output)
    feature_vector = extract_feature_model.predict(query_image) 

    print(feature_vector)
    # TODO
    # combine VGG16  and shallow layer output
    # shallow.add(Dense(4096))
    # shallow.add(BatchNormalization(axis=-1, momentum=0.99,epsilon=0.001))
    # shallow.add(Dense(512))

    # print(shallow.summary())
    # exit(0)

    # ## we can use batch normaliaztion from keras with axis specifies as across channels 
    # # shallow.add(LRN(name='shallow_norm', alpha=8191, n=8191, beta=0.5))

    # return model

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



# def build_model(self):

    ## read images from the triplets
    # for image_path in query_image:
    #     img = image.load_img(image_path, target_size=(224,224))
    #     img_data = image.img_to_array(img)
    #     img_data = np.expand_dims(img_data, axis=0)







#     query_input = Input((224, 224, 3))
#     positive_input = Input((224, 224, 3))
#     negative_input = Input((224, 224, 3))

#     query_embedding = self.build_base_network(query_input)
#     positive_embedding = self.build_base_network(positive_input)
#     negative_embedding = self.build_base_network(negative_input)

#     loss = merge([query_embedding, positive_embedding, negative_embedding],
#                  mode=triplet_loss, output_shape=(1,))
#     model = Model(inputs=[query_input, positive_input, negative_input],
#                   outputs=loss)

# def triplet_loss(self, X):

#     query, positive, negative = X

#     pos_dist = tf.reduce_sum(tf.square(tf.subtract(query, positive)), 1)
#     neg_dist = tf.reduce_sum(tf.square(tf.subtract(query, negative)), 1)

#     basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
#     loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

#     return loss


# TODO
# tune batch size
def create_batch(triplet_csv_path, batch_number, batch_size=20):

    query_image = np.zeros((batch_size,224,224,3))
    negative_image = np.zeros((batch_size,224,224,3))
    positive_image = np.zeros((batch_size,224,224,3))

    with open(triplet_csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in itertools.islice(csv_reader, batch_number*batch_size, (batch_number+1)*batch_size):
            # print(batch_number)
            print(count)
            # for every 20 rows/inputs
            # # model = VGG16(weights='imagenet', include_top=False)
            # # model.summary()



            query_img_path = row[0]
            print(row[0])
            query_img = image.load_img(query_img_path, target_size=(224, 224))
            query_img_data = image.img_to_array(query_img)
            query_img_data = np.expand_dims(query_img_data, axis=0)
            query_img_data = preprocess_input(query_img_data)
            
            query_image[count] = query_img_data
            # print(query_image.shape)
            # exit(0)



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



triplet_csv_path = 'triplets_skirts_10_sample_new2.csv'
batch_size = 16
# read images from triplets
total_train_size = 512 # update it to the shape of the data


print("Start training")
## testing purpose -- run only 1 batch
#for batch_num in range(32):
#    triplet_batch = create_batch(triplet_csv_path, batch_num, batch_size)
#    build_base_network(triplet_batch)


from keras.models import load_model
model = load_model('./model/fashion_lens_triplet_model.h5', custom_objects= {'triplet_loss':triplet_loss,'accuracy':accuracy, 'l2Norm':l2Norm, 'euclidean_distance':euclidean_distance})
print("Model loaded")

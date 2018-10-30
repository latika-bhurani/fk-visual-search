import keras.backend as K

"""
Utility functions 
"""

# Triplet loss is calculated on the basis of hinge loss
# with the following formula :
#           loss = max(0, _alpha + D(q, p) - D(q, n)
#           where D is the euclidean distance between the two images
# def triplet_loss(_, X, y_pred, _alpha):
#
#     query, positive, negative = X
#     positive_dist = K.square(query - positive)
#     negative_dist = K.square(query - negative)
#
#     # euclidean distance
#     positive_dist = K.sqrt((K.mean(positive_dist, axis = -1, keepdims=True)))
#     negative_dist = K.sqrt((K.mean(negative_dist, axis=-1, keepdims=True)))
#
#     loss = K.mean(K.maximum(0, _alpha + positive_dist - negative_dist))
#     return loss
#     # return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + _alpha))

def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(_, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
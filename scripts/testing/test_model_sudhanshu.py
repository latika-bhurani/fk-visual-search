from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input, Merge
from keras.layers.core import Dense
from keras.layers.merge import concatenate

# a single input layer
inputs = Input(shape=(3,))

# model 1
x1 = Dense(3, activation='relu')(inputs)
x1 = Dense(2, activation='relu')(x1)
x1 = Dense(2, activation='tanh')(x1)

print(x1.shape)
print(x1)
# model 2 
x2 = Dense(3, activation='linear')(inputs)
x2 = Dense(4, activation='tanh')(x2)
x2 = Dense(3, activation='tanh')(x2)

print(x2.shape)
print(x2)


# merging models
x3 = concatenate([x1, x2])

print(x3.shape)
print(x3)

exit(0)

# output layer
predictions = Dense(1, activation='sigmoid')(x3)

# generate a model from the layers above
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Always a good idea to verify it looks as you expect it to 
# model.summary()

data = [[1,2,3], [1,1,3], [7,8,9], [5,8,10]]
labels = [0,0,1,1]
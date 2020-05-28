# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:01:02 2020

@author: INSPIRON 3543
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metadata = pd.read_csv(r'F:\\562468_1022626_bundle_archive\\Chest_xray_Corona_Metadata.csv')

# creating training and testing data

train_data = metadata[metadata['Dataset_type'] == 'TRAIN']
train_data.drop(['Unnamed: 0'], inplace = True, axis = 1)

test_data = metadata[metadata['Dataset_type'] == 'TEST']
test_data.drop(['Unnamed: 0'], inplace = True, axis = 1)

#Filling NaN values

train_filled = train_data.fillna('Unknown')
test_filled = test_data.fillna('Unknown')

# Image augmentation

from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1./255., shear_range= 0.2, zoom_range=0.2,vertical_flip=True, validation_split = 0.25)
batch_size = 32
# Training,validating and testing data generator

train_generator = train_gen.flow_from_dataframe(dataframe=train_filled, 
                                                directory=r'F:\562468_1022626_bundle_archive\Coronahack-Chest-XRay-Dataset\Coronahack-Chest-XRay-Dataset\train',
                                                x_col="X_ray_image_name",
                                                y_col="Label",
                                                batch_size=batch_size,
                                                shuffle=True,
                                                subset = "training",
                                                class_mode="categorical",
                                                target_size = (128,128))
valid_generator = train_gen.flow_from_dataframe(dataframe=train_filled, 
                                                directory=r'F:\562468_1022626_bundle_archive\Coronahack-Chest-XRay-Dataset\Coronahack-Chest-XRay-Dataset\train',
                                                x_col="X_ray_image_name",
                                                y_col="Label",
                                                batch_size=batch_size,
                                                shuffle=True,
                                                subset = "validation",
                                                class_mode="categorical",
                                                target_size = (128,128))


test_gene = ImageDataGenerator(rescale=1./255.)

test_generator = test_gene.flow_from_dataframe(dataframe=test_filled,
                                               directory=r'F:\562468_1022626_bundle_archive\Coronahack-Chest-XRay-Dataset\Coronahack-Chest-XRay-Dataset\test',
                                               x_col="X_ray_image_name",
                                               y_col=None,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               class_mode=None,
                                               target_size=(128,128))

# CNN model - 3 Conv layers

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(64, (3,3),
                 input_shape = (128,128,3),
                 activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3),
                 activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3),
                 activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

train_step_size = train_generator.n//train_generator.batch_size
valid_step_size = valid_generator.n//valid_generator.batch_size
test_step_size = test_generator.n//test_generator.batch_size

# fitting

model.fit_generator(generator = train_generator,
                    steps_per_epoch = train_step_size,
                    validation_data = valid_generator,
                    validation_steps = valid_step_size,
                    epochs = 10)

# predicting
test_generator.reset()
pred = model.predict(test_generator, steps = test_step_size,verbose = 1)

predicted_classes=  np.argmax(pred, axis = 1)

labels = (train_generator.class_indices)
labels = dict((l,v) for v,l in labels.items())
prediction = [labels[v] for v in predicted_classes]


# testing image
from keras.preprocessing import image

test_image = image.load_img(r'F:\562468_1022626_bundle_archive\Coronahack-Chest-XRay-Dataset\Coronahack-Chest-XRay-Dataset\test\IM-0075-0001.jpeg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image2 = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image2)
train_generator.class_indices
if result[0][0] == 1:
    prediction = 'Normal'
else:
    prediction = 'Pnemonia'
    
fig= plt.figure(figsize = (10,10))
fig.suptitle(prediction)
plt.imshow(test_image[0], cmap = 'Greys')
plt.axis('on')
plt.colorbar()
plt.show()
    



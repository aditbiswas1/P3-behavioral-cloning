
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.utils import np_utils
from keras.regularizers import l2
from keras.preprocessing import image
from keras.optimizers import Adam
from keras import backend as K

data = pd.read_csv('data/driving_log.csv')
data_left = data.copy()[['left', 'steering']]
data_left['steering'] = data_left['steering'] + 0.22

data_right = data.copy()[['right', 'steering']]
data_right['steering'] = data_right['steering'] - 0.22

data_center = data.copy()[['center', 'steering']]

data_left.rename(columns={'left': 'image'}, inplace=True)
data_right.rename(columns={'right': 'image'}, inplace=True)
data_center.rename(columns={'center': 'image'}, inplace=True)



combined_data = pd.concat([data_center, data_left, data_right])
flipped_data = combined_data.copy()
combined_data['flip'] = False
flipped_data['flip'] = True
flipped_data['steering'] = flipped_data['steering'] * -1

final_data = pd.concat([combined_data, flipped_data, combined_data.copy(), flipped_data.copy()])


def crop_image(image, top=60, bottom=135):
    return image[top:bottom]


def resize_image(image):
    return cv2.resize(image,(224, 49), interpolation= cv2.INTER_AREA)


def augment_brightness(image):
    change_pct = np.random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness




def random_bright_augment(image, angle):
    choice = np.random.randint(2)
    if choice == 1:
        return augment_brightness(image), angle
    else:
        return image, angle


def training_augmentation_pipeline(entry):
    data_directory = 'data/'
    image_flip, angle = entry
    image = plt.imread(data_directory+image_flip[1]['image'].strip())
    angle = angle[1]['steering']
    image = resize_image(crop_image(image))
    image, angle = random_bright_augment(image, angle)
    
    flip = image_flip[1]['flip']
    if flip:
        image = image[:,:,::-1]

    return image, angle


images = final_data[['image', 'flip']]
angles = final_data[['steering']]


X_train, X_valid, y_train, y_valid = train_test_split(images, angles, test_size=0.1)


def training_image_generator(X, y, batch_size=128):
    N = X.shape[0]
    number_of_batches = int(np.ceil(N / batch_size))
    while True:
        X, y  = shuffle(X, y)
        for i in range(number_of_batches):
            start_index = i*batch_size
            end_index = (i+1)*(batch_size)
            if end_index <= N:
                X_batch = X[start_index:end_index]
                y_batch = y[start_index:end_index]
            else:
                X_batch = X[start_index:]
                y_batch = y[start_index:]
            
            X_batch, y_batch = X_batch.iterrows(), y_batch.iterrows()
            X_image_batch, y_batch = zip(*map(training_augmentation_pipeline, zip(X_batch, y_batch)))
            X_image_batch = np.asarray(X_image_batch)
            y_batch = np.asarray(y_batch)
            yield X_image_batch, y_batch
    

train_gen = training_image_generator(X_train, y_train)
valid_gen = training_image_generator(X_valid, y_valid)


model = Sequential([
        Lambda(lambda x: (x/ 127.5 - 1.),input_shape=(49,224,3)),
        Convolution2D(3, 1,1, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Convolution2D(32, 5 , 5, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Convolution2D(64, 5 , 5, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Convolution2D(64, 3 , 3, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(20, activation='tanh'),
        Dropout(0.2),
        Dense(1),
    ])
model.compile(optimizer=Adam(), loss='mse')
print(model.summary())

model.fit_generator(train_gen, X_train.shape[0], nb_epoch=10, validation_data=valid_gen, nb_val_samples=X_valid.shape[0])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

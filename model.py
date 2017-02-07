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


'''
preprocessing and augmenting the dataset:
in this section the entire log file is loaded into memory, this does not take up  much memory 
as it only contains strings for the image files and the steering angles / speed / throttle
'''

# make 3 data frames for left, center and right images alog with angle changes
data = pd.read_csv('data/driving_log.csv')
data_left = data.copy()[['left', 'steering']]
data_left['steering'] = data_left['steering'] + 0.22

data_right = data.copy()[['right', 'steering']]
data_right['steering'] = data_right['steering'] - 0.22

data_center = data.copy()[['center', 'steering']]

data_left.rename(columns={'left': 'image'}, inplace=True)
data_right.rename(columns={'right': 'image'}, inplace=True)
data_center.rename(columns={'center': 'image'}, inplace=True)


# join the 3 dataframes into 1 with only images and their corresponding angles
combined_data = pd.concat([data_center, data_left, data_right])
# setting flip flags on a copy of the dataframe to augment the hataset with horizontal flips
flipped_data = combined_data.copy()
combined_data['flip'] = False
flipped_data['flip'] = True
flipped_data['steering'] = flipped_data['steering'] * -1

# multiple copies of the dataset are concatenated, this was done to allow the images to be 
# augmented for brightness but make sure that the training epochs see enough examples.
# at this point the the dataframe contains ~90,000 image sources
final_data = pd.concat([combined_data, flipped_data, combined_data.copy(), flipped_data.copy()])


# function to crop image and remove scenery and car bumper
def crop_image(image, top=60, bottom=135):
    return image[top:bottom]

# function to resize images to acceptable size
def resize_image(image):
    return cv2.resize(image,(224, 49), interpolation= cv2.INTER_AREA)


# increate or decrease the brightness of image based on uniform probibility
def augment_brightness(image):
    change_pct = np.random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness




# apply brighness augmentation based on coin toss
def random_bright_augment(image, angle):
    choice = np.random.randint(2)
    if choice == 1:
        return augment_brightness(image), angle
    else:
        return image, angle


# compose the function to preprocess entries of a batch of image / angle / flip 
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


# split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(images, angles, test_size=0.1)


# generator function used for loading dataset from files in batches to avoid high memory consumption

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
    

# create generators to create training and validation batches
train_gen = training_image_generator(X_train, y_train)
valid_gen = training_image_generator(X_valid, y_valid)


# chosen model defintion 
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
        Dense(10, activation='tanh'),
        Dropout(0.2),
        Dense(1),
    ])
model.compile(optimizer=Adam(), loss='mse')
print(model.summary())

# training model using my defined generators
model.fit_generator(train_gen, X_train.shape[0], nb_epoch=10, validation_data=valid_gen, nb_val_samples=X_valid.shape[0])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

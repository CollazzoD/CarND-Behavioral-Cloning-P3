import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import os

CSV_FILENAME = 'driving_log.csv'
STEERING_CORRECTION = 0.2
LOG_COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'left_steering', 'right_steering']
BATCH_SIZE = 32
N_EPOCHS = 3

# Search for all the file 'driving_log.csv' inside a given root folder
def find_all_driving_log(search_root):
    global CSV_FILENAME
    logs_path = []
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file == CSV_FILENAME:
                logs_path.append(root)
    return logs_path

# This function opens the csv file containing images path and steerings,
# applies correction for left and right images steerings and return a list
# of couples {image, steering} where image is the image's path and steering
# the corresponding steering value
def get_images_and_steerings(data_filepath, columns = None):
    global CSV_FILENAME, STEERING_CORRECTION, LOG_COLUMNS
    # Read 'driving_log.csv'
    if columns:
        df_img = pd.read_csv(data_filepath + '/' + CSV_FILENAME, names = LOG_COLUMNS)
    else:
        df_img = pd.read_csv(data_filepath + '/' + CSV_FILENAME)
        
    df_img['center'] = data_filepath + '/IMG/' + df_img['center'].str.split('/').str[-1].str.strip()
    df_img['left'] = data_filepath + '/IMG/' + df_img['left'].str.split('/').str[-1].str.strip()
    df_img['right'] = data_filepath + '/IMG/' + df_img['right'].str.split('/').str[-1].str.strip()
    df_img['left_steering'] = df_img['steering'] + STEERING_CORRECTION
    df_img['right_steering'] = df_img['steering'] - STEERING_CORRECTION

    samples = []
    center_samples = df_img[['center', 'steering']].rename(columns = {'center':'image'}).to_dict('records')
    left_samples = df_img[['left', 'left_steering']].rename(columns = {'left':'image', 'left_steering':'steering'}).to_dict('records')
    right_samples = df_img[['right', 'right_steering']].rename(columns = {'right':'image', 'right_steering':'steering'}).to_dict('records')

    samples.extend(center_samples)
    samples.extend(left_samples)
    samples.extend(right_samples)

    return samples

# Creates NVIDIA Model 
# Creates NVIDIA Model 
def NVIDIA_Model():
    model = Sequential()
    # Normalize data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crops images
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    # Add NVIDIA Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model

# Generator used to improve memory efficiency while training the model
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            for batch_sample in batch_samples:
                img_path = batch_sample['image']
                image = cv2.imread(img_path)
                steering = float(batch_sample['steering'])
                images.append(image)
                steerings.append(steering)

                # Flip the image in order to augment the dataset
                images.append(cv2.flip(image,1))
                steerings.append(steering * -1.0)

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

if __name__ == '__main__':
    global LOG_COLUMNS
    
    # Get images from course's dataset 
    samples = get_images_and_steerings('./data')

    # Search for recorded datasets
    for log_path in find_all_driving_log('../rec'):
        s = get_images_and_steerings(log_path, columns = LOG_COLUMNS)
        samples.extend(s)

    print("Number of images (center + left + right) ", len(samples))
    print("Number of images used to train the network (center + left + right) * 2 (flipping)", len(samples) * 2)

    # Split all samples in train and validation sets
    train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

    print("Number of train samples ", len(train_samples))
    print("Number of validation samples ", len(validation_samples))

    # Define the train generator and the validation generator in order to
    # ease memory occupation
    train_generator = generator(train_samples, batch_size = BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

    # Define the model
    model = NVIDIA_Model()

    # Compile the model
    model.compile(loss = 'mse', optimizer = 'adam')

    # Train the model and get the history in order to visualize the loss (as seen in the course)
    # Modified after reading https://knowledge.udacity.com/questions/14296
    history_object = model.fit_generator(train_generator, 
        validation_data = validation_generator,          
        steps_per_epoch = np.ceil( len(train_samples) / BATCH_SIZE), 
        verbose = 1,
        epochs = N_EPOCHS,
        validation_steps = np.ceil( len(validation_samples) / BATCH_SIZE)     
    )

    # Plot training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()

    print('Training Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

    
    # Save the model
    model.save('model.h5')
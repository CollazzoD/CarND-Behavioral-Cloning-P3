import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

learning_rate = 0.0001
epochs = 20

# This function opens the csv file containing images path and steerings,
# applies correction for left and right images steerings and return a list
# of couples {image, steering} where image is the image's path and steering
# the corresponding steering value
def get_images_and_steerings():
    DATA_FILEPATH = './data'
    CSV_FILENAME = 'driving_log.csv'
    STEERING_CORRECTION = 0.2
    df_img = pd.read_csv(f'{DATA_FILEPATH}/{CSV_FILENAME}')
    df_img['left_steering'] = df_img['steering'] + STEERING_CORRECTION
    df_img['right_steering'] = df_img['steering'] - STEERING_CORRECTION

    samples = []
    center_samples = df_img[['center', 'steering']].rename(columns = {'center':'image'}).to_dict('records')
    left_samples = df_img[['left', 'left_steering']].rename(columns = {'left':'image', 'left_steering':'steering'}).to_dict('records')
    right_samples = df_img[['right', 'right_steering']].rename(columns = {'right':'image', 'right_steering':'steering'}).to_dict('records')

    samples.extend(center_samples, left_samples, right_samples)

    return samples

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

# Code taken from course's video
images = []
measurements = []
csv_filepath = '../data/driving_log.csv'
with open(csv_filepath) as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # # create adjusted steering measurements for the side camera images
        # correction = 0.2 # this is a parameter to tune
        # steering_left = steering_center + correction
        # steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = '../data/IMG/' # fill in the path to your training IMG directory
        img_center = process_image(np.asarray(Image.open(path + row[0])))
        img_left = process_image(np.asarray(Image.open(path + row[1])))
        img_right = process_image(np.asarray(Image.open(path + row[2])))

        # add images and angles to data set
        car_images.extend(img_center, img_left, img_right)
        steering_angles.extend(steering_center, steering_left, steering_right)


for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path =  + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

# Flip the images in order to augment dataset
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')


# End of code taken from course's video
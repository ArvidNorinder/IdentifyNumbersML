import tensorflow as tf
import numpy as np
import cv2
import os

loaded_model = tf.keras.models.load_model('mnist.h5')

folder_path = "./images/"

image_list = []
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Specify the image file extensions you want to consider
        image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
        if image is not None:
            image = cv2.resize(image, (28, 28))
            image_list.append(image)

# Convert the list of images to a numpy array
image_array = np.array(image_list)

print(image_array.shape)
# Reshape the array to 4 dimensions (number of images, height, width, number of channels)
image_array = image_array.reshape(image_array.shape[0], 28, 28, 1)

# Normalize the pixel values to values between 0 and 1
image_array = image_array.astype('float32')
image_array /= 255

# Predict the number in the image
predictions = loaded_model.predict(image_array)

# Print the predictions
print(predictions)

# Print the number in the image
print(np.argmax(predictions, axis=1))

# Print the probabilities of each number
print(np.max(predictions, axis=1))
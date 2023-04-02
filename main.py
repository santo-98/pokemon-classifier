import os
import cv2
import tensorflow as tf
from keras import layers, models
from keras.models import load_model
import numpy as np

# NOTE:I know this code smells. Currently i am exhausted from making the model work. Will clean it up later
def prepare(filepath):
  IMG_SIZE = 256
  img_array = cv2.imread(filepath)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0

def train_and_save():
  normalization_layer = layers.Rescaling(1./255)
  data = tf.keras.utils.image_dataset_from_directory("data")
  class_names = data.class_names
  data = data.map(lambda x, y: (normalization_layer(x), y))
  data.as_numpy_iterator().next()

  train_size = int(len(data)*.8)
  val_size = int(len(data)*.2)

  train = data.take(train_size)
  val = data.skip(train_size).take(val_size)

  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256,256, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(len(class_names)))

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.fit(train, epochs=5, verbose=True, validation_data=val)

  model.save('imageclassifier.h5')


def predict(path):
  model = load_model('imageclassifier.h5')
  class_names = sorted(os.listdir('data'))
  img = tf.keras.utils.load_img(
      path, target_size=(256, 256)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  prediction = model.predict(img_array)
  print("This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(tf.nn.softmax(prediction[0]))], 100 * np.max(tf.nn.softmax(prediction[0]))))

predict('test_img.jpg')

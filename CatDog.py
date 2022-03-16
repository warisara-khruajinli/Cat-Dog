import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator

base_dir = './train'
dog_dir = os.path.join(base_dir, 'dog')
cat_dir = os.path.join(base_dir, 'cat')

dog_df = pd.DataFrame(os.listdir(dog_dir), columns=["filename"])
dog_df["filename"] = 'dog/'+dog_df["filename"]
dog_df["class"] = "dog"
cat_df = pd.DataFrame(os.listdir(cat_dir), columns=["filename"])
cat_df["filename"] = 'cat/'+cat_df["filename"]
cat_df["class"] = "cat"

train_size = 0.9

dog_train, dog_test = train_test_split(
    dog_df, train_size=train_size, random_state=1234)
cat_train, cat_test = train_test_split(
    cat_df, train_size=train_size, random_state=1234)

dog_train, dog_validate = train_test_split(
    dog_train, test_size=1/6, random_state=1234)
cat_train, cat_validate = train_test_split(
    cat_train, test_size=1/6, random_state=1234)

train_df = pd.concat([dog_train, cat_train], ignore_index=True)
validate_df = pd.concat([dog_validate, cat_validate], ignore_index=True)
test_df = pd.concat([dog_test, cat_test], ignore_index=True)

train_df

IMAGE_SIZE = 150
BATCH_SIZE = 120

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=60,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.1,
                                           zoom_range=0.2,
                                           fill_mode='nearest')

train_data = train_image_generator.flow_from_dataframe(
    train_df,
    directory=base_dir,
    x_col="filename",
    y_col="class",
    class_mode="binary",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

validate_image_generator = ImageDataGenerator(rescale=1./255)

validate_data = validate_image_generator.flow_from_dataframe(
    validate_df,
    directory=base_dir,
    x_col="filename",
    y_col="class",
    class_mode="binary",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

model = tf.keras.models.Sequential([
    # CNN Layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    # ANN Layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    metrics=['accuracy'])
model.summary()

EPOCHS = 5
history = model.fit(train_data, epochs=EPOCHS,
                    validation_data=validate_data, batch_size=BATCH_SIZE)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

class_labels = ["cat", "dog"]


def showImage(img):
    plt.imshow(img)
    plt.show()


def predict_img(img_path, show=False):
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255
    img_array = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    predictions = model.predict(img_array)
    scores = predictions[0][0]
    if scores >= 0.5:
        pred_class = "dog"
    else:
        pred_class = "cat"
    if show:
        showImage(img_array[0])
        print(scores)
    return pred_class

predict_img('train/cat/cat.1560.jpg',True)

model.save("model.h5")
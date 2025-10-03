!pip install fastapi

# ---- New Cell ----

!pip install uvicorn
!pip install python-multipart
!pip install pillow

# ---- New Cell ----

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# ---- New Cell ----

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3

# ---- New Cell ----

dataset =  tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle= True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
) 

# ---- New Cell ----

class_names = dataset.class_names
class_names

# ---- New Cell ----

len(dataset)

# ---- New Cell ----

for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy().shape)

# ---- New Cell ----

train_size = 0.8
train_ds= dataset.take(54)
len(train_ds)

# ---- New Cell ----

test_ds = dataset.skip(54)
len(test_ds)


# ---- New Cell ----

val_ds= test_ds.take(6)
len(val_ds)

# ---- New Cell ----

test_ds = test_ds.skip(6)
len(test_ds)

# ---- New Cell ----

def get_dataset_partitions_tf(ds, train_split=0.8, val_split = 0.1, test_split= 0.1, shuffle= True, shuffle_size =10000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
        
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)

    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

# ---- New Cell ----

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# ---- New Cell ----

train_ds =train_ds.cache().shuffle (1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds =val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds =test_ds.cache().shuffle (1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# ---- New Cell ----

from tensorflow.keras import layers

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])


# ---- New Cell ----

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# ---- New Cell ----

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),  
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(n_classes, activation='softmax')
])


# ---- New Cell ----

model.summary()

# ---- New Cell ----

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# ---- New Cell ----

EPOCHS=50
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

# ---- New Cell ----

scores = model.evaluate(test_ds)

# ---- New Cell ----

scores


# ---- New Cell ----

history


# ---- New Cell ----

history.params

# ---- New Cell ----

history.history.keys()

# ---- New Cell ----

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss =history.history['loss']
val_loss =history.history['val_loss']

# ---- New Cell ----

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range (EPOCHS), acc, label='Training Accuracy')
plt.plot(range (EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# ---- New Cell ----

import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    first_image= images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0]

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])
    
    batch_prediction= model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])

# ---- New Cell ----

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions =model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 *(np.max(predictions[0])), 2)
    return predicted_class, confidence


# ---- New Cell ----

plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")
        

# ---- New Cell ----

model_version=1
model.save(f"C:/Users/Rohan Patil/potato-disease/models/{model_version}.keras")



# ---- New Cell ----


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Directory paths
train_dir = r'C:\Users\wt\Desktop\emotionDataset\train'
test_dir = r'C:\Users\wt\Desktop\emotionDataset\test'

# Image data generators with grayscale to RGB conversion and resizing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of data for validation
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

def preprocess_images(generator):
    while True:
        images, labels = next(generator)
        rgb_images = np.stack([images[:, :, :, 0]] * 3, axis=-1)  # Convert grayscale to RGB
        yield rgb_images, labels

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224
    color_mode='grayscale',  # Read as grayscale
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224
    color_mode='grayscale',  # Read as grayscale
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Resize images to 224x224
    color_mode='grayscale',  # Read as grayscale
    batch_size=32,
    class_mode='categorical'
)

# Create the MobileNet model
def create_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Train the model
history = model.fit(
    preprocess_images(train_generator),
    epochs=10,
    validation_data=preprocess_images(validation_generator),
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(preprocess_images(test_generator), steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()

# Visualize sample images
def visualize_samples(generator, num_samples=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        img, label = next(generator)
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img[0])
        plt.title(generator.class_indices)
        plt.axis('off')
    plt.show()

visualize_samples(validation_generator)


# In[ ]:





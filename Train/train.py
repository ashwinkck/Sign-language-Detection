import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json


data_dir = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data'

# Set parameters
img_height, img_width = 128, 128  
batch_size = 32
epochs = 20

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  
)

# Load the pre-trained MobileNetV2 model without the top classification layer
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensionality
output = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer for classification

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers to prevent their weights from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the model to a file
model.save('sign_language_model.h5')  # Corrected extension

print("Model saved as sign_language_model.h5")

# Save class names
class_indices = train_generator.class_indices  
class_names = list(class_indices.keys())  

# Save the class names to a JSON file
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Class names saved to class_names.json")

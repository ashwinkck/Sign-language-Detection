import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# Set dataset directory
data_dir = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data'  # Ensure correct path

# Specify the letters you want to train on
target_letters = ['d', 'e', 'b']  # Replace with desired letters

# Create a new directory for the selected letters
filtered_data_dir = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Filtered_Data'
if not os.path.exists(filtered_data_dir):
    os.makedirs(filtered_data_dir)

# Copy only the folders for the specified letters
for letter in target_letters:
    source_folder = os.path.join(data_dir, letter)
    if os.path.exists(source_folder):
        os.makedirs(os.path.join(filtered_data_dir, letter), exist_ok=True)
        # Copy images from the source folder to the filtered folder
        for filename in os.listdir(source_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
                os.rename(os.path.join(source_folder, filename),
                          os.path.join(filtered_data_dir, letter, filename))

# Set parameters
img_height, img_width = 128, 128  # Image dimensions
batch_size = 32
epochs = 15

# Create data generators with augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split data into training and validation sets
)

# Load training and validation data from the filtered directory
train_generator = datagen.flow_from_directory(
    filtered_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

val_generator = datagen.flow_from_directory(
    filtered_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
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
model.save('sign_language_model.h5')

print("Model saved as sign_language_model.h5")

# Save class names
class_indices = train_generator.class_indices  # Get the class indices (dictionary)
class_names = list(class_indices.keys())  # Get the list of class names

# Save the class names to a JSON file
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Class names saved as class_names.json")

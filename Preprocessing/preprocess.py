import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Directory path where your images are stored (organized by class folders, e.g., 'A', 'B', etc.)
data_dir =  r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data'
# Image dimensions and target classes
img_height, img_width = 64, 64
class_names = os.listdir(data_dir)  # Get folder names (assumed as class names)
num_classes = len(class_names)

# Preprocessing function: Resize and normalize images
def preprocess_image(image_path):
    # Read image from file
    img = cv2.imread(image_path)
    
    # Resize image to the target size
    img = cv2.resize(img, (img_height, img_width))
    
    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    return img

# Load images and labels
images = []
labels = []

for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = preprocess_image(img_path)
        
        images.append(img)
        labels.append(idx)  # Use the index as the label (corresponding to the class name)

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# One-hot encoding the labels for categorical classification
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Data augmentation using ImageDataGenerator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Augment training data
datagen.fit(X_train)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=20, 
                    validation_data=(X_val, y_val))

# Save the model
model.save('sign_language_model.h5')
print("Model saved as sign_language_model.xh5")

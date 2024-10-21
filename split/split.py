import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set image dimensions and parameters
img_height, img_width = 64, 64  # Change this to match your dataset
batch_size = 32
num_classes = 26  # Assuming 26 sign language alphabet classes

# ImageDataGenerator with validation split
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest',
    validation_split=0.2  # Automatically split 20% of the data for validation
)

# Load your training and validation data from the same directory, but using the split
train_data = datagen.flow_from_directory(
    r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data',  # Path to your data folder containing all images
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Training data subset
)

val_data = datagen.flow_from_directory(
    r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data',  # Path to your data folder containing all images
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Validation data subset
)

# Define the CNN model architecture
model = models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Dropout for regularization
model.add(layers.Dropout(0.3))  # Dropout with 30% of neurons randomly disabled

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))

# Add another Dropout
model.add(layers.Dropout(0.5))  # Dropout with 50%

# Output layer (Softmax for multi-class classification)
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set up callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
epochs = 50  # You can adjust the number of epochs
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

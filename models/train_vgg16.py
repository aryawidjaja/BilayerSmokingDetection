import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, AveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from plot_utils import plot_training_history
from custom_callbacks import CustomEarlyStopping
import matplotlib.pyplot as plt

# Define the base path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths
data_dir = os.path.join(base_path, 'data', 'new_smoking_images')
train_path = os.path.join(data_dir, 'Training')
val_path = os.path.join(data_dir, 'Validation')
test_path = os.path.join(data_dir, 'Testing')

# Image Data Generator setup with augmentation
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

# Load the VGG16 model with pretrained weights
vgg16 = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Add new layers for fine-tuning
for layer in vgg16.layers:
    layer.trainable = False

x = vgg16.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten(name="flatten")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=vgg16.input, outputs=predictions)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Callbacks
early_stopping = CustomEarlyStopping(train_acc_threshold=0.80, val_acc_threshold=0.70)
model_checkpoint = ModelCheckpoint('../models/best_vgg16_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot training and validation loss and accuracy
plot_training_history(history)

# Save the final model
model.save('../models/smoking_detection_vgg16_model.keras')

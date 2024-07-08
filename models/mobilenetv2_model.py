import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from glob import glob
import warnings

from plot_utils import plot_training_history, plot_confusion_matrix, display_classification_report, plot_actual_vs_predicted_images

warnings.filterwarnings('ignore')

# Custom early stopping callback
class CustomEarlyStopping(Callback):
    def __init__(self, train_acc_threshold=0.80, val_acc_threshold=0.70):
        super(CustomEarlyStopping, self).__init__()
        self.train_acc_threshold = train_acc_threshold
        self.val_acc_threshold = val_acc_threshold

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc is not None and val_acc is not None:
            if train_acc > self.train_acc_threshold and val_acc > self.val_acc_threshold:
                print(f"\nStopping training as training accuracy {train_acc:.4f} > {self.train_acc_threshold} "
                      f"and validation accuracy {val_acc:.4f} > {self.val_acc_threshold}")
                self.model.stop_training = True

# Function to process each dataset
def process_dataset(df, path, label_list):
    img_list = glob(path + '/*.jpg')
    for img in img_list:
        file_name = os.path.splitext(img)[0].split("/")[-1]
        for i, label in enumerate(label_list):
            if file_name.startswith(label):
                new_data = pd.DataFrame({"path": [img], "label": [label], "class_id": [i]})
                df = pd.concat([df, new_data], ignore_index=True)
                break
    df[["path"]] = df[["path"]].astype(str)
    df[["label"]] = df[["label"]].astype(str)
    df[["class_id"]] = df[["class_id"]].astype(int)
    return df

# Paths
train_path = '../data/smoking_images/Training'
val_path = '../data/smoking_images/Validation'
test_path = '../data/smoking_images/Testing'
label_list = ['notsmoking', 'smoking']

# Process datasets
train_df = process_dataset(pd.DataFrame({"path": [], "label": [], "class_id": []}), train_path, label_list)
val_df = process_dataset(pd.DataFrame({"path": [], "label": [], "class_id": []}), val_path, label_list)
test_df = process_dataset(pd.DataFrame({"path": [], "label": [], "class_id": []}), test_path, label_list)

# Image Data Generator setup with augmentation
img_size = (224, 224)
batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_dataframe(train_df, x_col='path', y_col='label', target_size=img_size,
                      class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

# Validation & Test Data Generator setup without augmentation
valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(val_df, x_col='path', y_col='label', target_size=img_size,
                                                     class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_df, x_col='path', y_col='label', target_size=img_size,
                                                    class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

# Get class indices dictionary
gen_dict = train_gen.class_indices
classes = list(gen_dict.keys())

# Build the MobileNetV2 model
input_shape = (224, 224, 3)
num_class = len(classes)
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'),  # Reduced size
    Dropout(0.5),  # Increased dropout rate
    BatchNormalization(),
    Dense(num_class, activation='softmax')
])

# Compile the model with a smaller learning rate
model.compile(Adamax(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# Commented out due to shape issues
# print("Model Summary:")
# model.summary()

# Set up callbacks
early_stopping = CustomEarlyStopping(train_acc_threshold=0.80, val_acc_threshold=0.70)
model_checkpoint = ModelCheckpoint('../models/best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(train_gen, epochs=50, verbose=1, validation_data=valid_gen, shuffle=False, callbacks=[early_stopping, model_checkpoint])

# Plot training and validation loss and accuracy
plot_training_history(history)

# Classification report and confusion matrix
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=-1)
class_names = list(test_gen.class_indices.keys())

# Display Classification Report
display_classification_report(y_true, y_pred, class_names)

# Plot Confusion Matrix
plot_confusion_matrix(y_true, y_pred, class_names)

# Plot actual vs predicted images
plot_actual_vs_predicted_images(model, test_gen)

# Save the final model
model.save('../models/smoking_detection_model_final.keras')

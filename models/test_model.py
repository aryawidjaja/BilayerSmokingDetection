import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from plot_utils import plot_confusion_matrix, display_classification_report, plot_actual_vs_predicted_images

# Define the base path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths
data_dir = os.path.join(base_path, 'data', 'new_smoking_images')
test_path = os.path.join(data_dir, 'Testing')

# Image Data Generator setup
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Load the trained model
model_path = os.path.join(base_path, 'models', 'best_vgg16_model.keras')
model = tf.keras.models.load_model(model_path)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predictions
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

# Display Classification Report
class_names = list(test_generator.class_indices.keys())
display_classification_report(y_true, y_pred, class_names)

# Plot Confusion Matrix
plot_confusion_matrix(y_true, y_pred, class_names)

# Plot actual vs predicted images
plot_actual_vs_predicted_images(model, test_generator)

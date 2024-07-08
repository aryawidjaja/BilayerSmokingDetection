import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def display_classification_report(y_true, y_pred, class_names):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_actual_vs_predicted_images(model, test_gen, num_images=10):
    class_names = list(test_gen.class_indices.keys())
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        images, labels = next(test_gen)
        predictions = model.predict(images)
        for j in range(len(images)):
            if (i * num_cols + j) < num_images:
                actual_label = class_names[int(labels[j])]
                predicted_label = class_names[int(predictions[j] > 0.5)]
                color = 'green' if actual_label == predicted_label else 'red'
                plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
                plt.imshow(images[j])
                plt.axis('off')
                plt.title(f'Actual: {actual_label}\nPredicted: {predicted_label}', color=color)
    plt.tight_layout()
    plt.show()

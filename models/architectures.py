import tensorflow as tf
from tensorflow import keras

def create_teacher_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create teacher model architecture (larger, more complex).
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled teacher model
    """
    model = keras.Sequential([
        keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes)
    ], name="TeacherCNN")
    
    return model

def create_student_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create student model architecture (smaller, simpler).
    """
    model = keras.Sequential([
        keras.layers.Conv2D(8, 5, activation='relu', input_shape=input_shape),  # Menos filtros
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),  # Menos neuronas
        keras.layers.Dense(num_classes)
    ], name="StudentCNN")
    
    return model

import tensorflow as tf
from tensorflow import keras
import os

class DistillationTrainer:
    """Main trainer class for knowledge distillation experiments."""
    
    def __init__(self, config):
        self.config = config
        self.teacher_model = None
        self.distiller = None
        
    def train_teacher(self, teacher_model, x_train, y_train, x_test, y_test):
        """
        Train the teacher model.
        
        Args:
            teacher_model: Teacher architecture
            x_train, y_train: Training data
            x_test, y_test: Test data
            
        Returns:
            Trained teacher model with accuracy
        """
        print("Training teacher model...")
        
        teacher_model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train teacher
        teacher_model.fit(
            x_train, y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['models']['teacher']['epochs'],
            validation_data=(x_test, y_test),
            verbose=self.config['output']['verbose']
        )
        
        # Evaluate teacher
        teacher_accuracy = teacher_model.evaluate(x_test, y_test, verbose=0)[1]
        print(f"Teacher model accuracy: {teacher_accuracy:.4f}")
        
        self.teacher_model = teacher_model
        return teacher_model, teacher_accuracy
    
    def train_distillation(self, distiller, x_train, y_train, x_test, y_test):
        """
        Train the student model using knowledge distillation.
        
        Args:
            distiller: KnowledgeDistiller instance
            x_train, y_train: Training data
            x_test, y_test: Test data
            
        Returns:
            Training history and final accuracy
        """
        print("Training student model with knowledge distillation...")
        
        # Train with knowledge distillation
        history = distiller.fit(
            x_train, y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(x_test, y_test),
            verbose=self.config['output']['verbose']
        )
        
        # Evaluate final model - Debug primero
        evaluation_result = distiller.evaluate(x_test, y_test, verbose=0)
        print(f"DEBUG - Evaluation result type: {type(evaluation_result)}")
        print(f"DEBUG - Evaluation result: {evaluation_result}")
        
        student_accuracy = float(evaluation_result[1]['sparse_categorical_accuracy'])
        print(f"Student model accuracy: {student_accuracy:.4f}")
        
        return history, student_accuracy
    
    def save_models(self, distiller, save_path):
        """Save the trained student model."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        student_path = os.path.join(save_path, 'student_model.h5')
        distiller.student.save(student_path)
        print(f"Student model saved to: {student_path}")

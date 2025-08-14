import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class ExperimentMetrics:
    """Utility class for tracking and visualizing experiment metrics."""
    
    def __init__(self):
        self.results = {}
    
    def log_results(self, teacher_acc, student_acc):
        """Log experiment results."""
        self.results = {
            'teacher_accuracy': teacher_acc,
            'student_accuracy': student_acc,
            'improvement': student_acc - 0.0  # Baseline comparison removed
        }
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS")
        print("="*50)
        print(f"Teacher accuracy:      {self.results['teacher_accuracy']:.4f}")
        print(f"Student (KD) accuracy: {self.results['student_accuracy']:.4f}")
        print("Knowledge distillation training completed successfully!")
    
    def test_prediction(self, model, x_test, y_test, sample_idx=0):
        """Test model prediction on a sample."""
        sample_image = x_test[sample_idx:sample_idx+1]
        prediction = model(sample_image)
        predicted_digit = tf.argmax(prediction, axis=1).numpy()[0]
        actual_digit = y_test[sample_idx]
        confidence = tf.nn.softmax(prediction)[0][predicted_digit].numpy()
        
        print(f"\nSample prediction test:")
        print(f"Actual digit: {actual_digit}")
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.3f}")
        
        return predicted_digit == actual_digit

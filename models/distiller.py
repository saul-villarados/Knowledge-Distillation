import tensorflow as tf
from tensorflow import keras

class KnowledgeDistiller(keras.Model):
    """
    Knowledge Distillation implementation for teacher-student learning.
    
    Args:
        student: Student model (smaller architecture)
        teacher: Teacher model (larger, pre-trained architecture)
        temperature: Temperature parameter for softmax scaling
        alpha: Weight balance between student loss and distillation loss
    """
    
    def __init__(self, student, teacher, temperature=4.0, alpha=0.1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher.trainable_variables:
            param.assign(param)
        self.teacher.trainable = False

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
        """Compile the distiller with loss functions and metrics."""
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        """Custom training step implementing knowledge distillation."""
        x, y = data

        # Get teacher predictions without gradients
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Get student predictions
            student_predictions = self.student(x, training=True)

            # Calculate student loss (standard cross-entropy)
            student_loss = self.student_loss_fn(y, student_predictions)

            # Calculate distillation loss with temperature scaling and T² factor
            teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature, axis=1)
            student_soft = tf.nn.softmax(student_predictions / self.temperature, axis=1)
            
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)
            distillation_loss *= (self.temperature ** 2)

            # Combine losses
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Update student model weights
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        return {
            "loss": total_loss,
            "student_loss": student_loss,
            "distillation_loss": distillation_loss
        }

    def test_step(self, data):
        """Custom test step for evaluation."""
        x, y = data
        student_predictions = self.student(x, training=False)
        self.compiled_metrics.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}

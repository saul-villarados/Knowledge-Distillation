import sys
import os
import yaml
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

from models.distiller import KnowledgeDistiller
from models.architectures import create_teacher_model, create_student_model
from src.data.loader import MNISTDataLoader
from src.training.trainer import DistillationTrainer
from src.utils.metrics import ExperimentMetrics

def load_config():
    """Load configuration from YAML file."""
    config_path = 'config\config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main experiment pipeline."""
    print("Starting Knowledge Distillation experiment on MNIST...")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader = MNISTDataLoader()
    trainer = DistillationTrainer(config)
    metrics = ExperimentMetrics()
    
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    data_loader.get_data_info(x_train, y_train, x_test, y_test)
    
    # Create models
    teacher_model = create_teacher_model()
    student_model = create_student_model()
    
    print(f"Teacher parameters: {teacher_model.count_params():,}")
    print(f"Student parameters: {student_model.count_params():,}")
    
    # Display model summaries
    print("\nTeacher Model Architecture:")
    teacher_model.summary()
    print("\nStudent Model Architecture:")
    student_model.summary()
    
    # Train teacher model
    teacher_model, teacher_acc = trainer.train_teacher(
        teacher_model, x_train, y_train, x_test, y_test
    )
    
    # Create and configure knowledge distiller
    distiller = KnowledgeDistiller(
        student=student_model,
        teacher=teacher_model,
        temperature=config['distillation']['temperature'],
        alpha=config['distillation']['alpha']
    )
    
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence()
    )
    
    print(f"KD Configuration:")
    print(f"  Temperature: {config['distillation']['temperature']}")
    print(f"  Alpha: {config['distillation']['alpha']}")
    
    # Train with knowledge distillation
    history, student_acc = trainer.train_distillation(
        distiller, x_train, y_train, x_test, y_test
    )
    
    # Save models
    trainer.save_models(distiller, config['output']['model_save_path'])
    
    # Log and display results
    metrics.log_results(teacher_acc, student_acc)
    metrics.print_summary()
    metrics.test_prediction(distiller.student, x_test, y_test)

if __name__ == "__main__":
    main()
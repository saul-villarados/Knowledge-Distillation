# 🧠 Knowledge Distillation for MNIST  

A modular implementation of **Knowledge Distillation** using **TensorFlow/Keras** for MNIST digit classification.  

## ✨ Features  
- 🏫 Teacher-Student architecture with CNN models  
- 🌡️ Configurable temperature and alpha parameters  
- 📊 Automatic model comparison and evaluation  
- 🧩 Clean, modular codebase  

## 🚀 Quick Start  
```bash
git clone https://github.com/yourusername/knowledge-distillation-mnist.git  
cd knowledge-distillation-mnist  
pip install -r requirements.txt  
python experiments/run_experiment.py  
```
📈 Results

The distilled student model achieved same performance as Teacher yielding:

- 📦 Improvement: ~3x Over Space Size

- ⚙️ Configuration

Edit config/config.yaml to modify:

- 🏗️ Model architectures

- 🔧 Training parameters

- 🔥 Distillation settings

📦 requirements.txt
```bash
tensorflow>=2.13.0  
numpy>=1.21.0  
pyyaml>=6.0  
matplotlib>=3.5.0  
```
🔥 Knowledge Distillation Parameters
```bash
distillation:  
  temperature: 7.0  
  alpha: 0.1   # Weight for student loss (1-alpha for distillation loss)  
```
🏗️ Model Configuration
```bash
models:  
  teacher:  
    name: "TeacherCNN"  
    epochs: 3  
  student:  
    name: "StudentCNN"  

```
💾 Output Configuration
```bash
output:  
  model_save_path: "models/"  
  verbose: 1  


```

# Cybersecurity Threat Detection System

A deep learning-based cybersecurity threat detection system using the BETH dataset to classify malicious vs benign system events.

## 🎯 Project Overview

This project implements a neural network that analyzes system event logs to automatically detect cybersecurity threats in real-time. The system achieves high accuracy in identifying malicious activities such as malware execution, privilege escalation, and system exploitation.

## 📊 Dataset

- **Source**: [BETH Dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset)
- **Features**: 7 numerical columns from system event logs
- **Target**: Binary classification (0=benign, 1=malicious)
- **Size**: 40MB+ of preprocessed cybersecurity data

## 🛠️ Technology Stack

- **Machine Learning**: PyTorch, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Evaluation**: torchmetrics, classification reports

## 🚀 Features

- **Advanced Neural Network**: 4-layer architecture with dropout
- **Smart Training**: Early stopping, learning rate scheduling
- **Comprehensive Evaluation**: Precision, recall, F1-score, confusion matrix
- **Professional Visualizations**: Training curves, performance charts
- **Production Ready**: Model checkpointing, L2 regularization

## 📈 Performance

- **High Accuracy**: Achieves excellent threat detection rates
- **Low False Positives**: Optimized for real-world deployment
- **Fast Training**: Efficient convergence with Adam optimizer
- **Robust**: Dropout and regularization prevent overfitting

## 🔍 How It Works

1. **Data Ingestion**: Loads system event logs with 7 features
2. **Preprocessing**: Standardizes features for optimal training
3. **Model Training**: Neural network learns threat patterns
4. **Evaluation**: Comprehensive performance analysis
5. **Visualization**: Professional charts and metrics

## 🎯 Real-World Applications

- **Security Operations Centers (SOCs)**
- **Intrusion Detection Systems (IDS)**
- **SIEM Platforms**
- **Endpoint Protection Solutions**

## 📁 Project Structure

```
├── cybersecurity_threat_detection.ipynb    # Main analysis notebook
├── accreditation.md       # Dataset attribution
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

## 🏆 Key Achievements

- ✅ Complete end-to-end ML pipeline
- ✅ Professional-grade visualizations
- ✅ Advanced training techniques
- ✅ Comprehensive evaluation metrics
- ✅ Production-ready architecture

## 🔬 Technical Details

### Model Architecture
```
Input (7 features) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.1) → Output(1)
```

### Training Features
- **Loss Function**: Binary Cross-Entropy Loss
- **Optimizer**: Adam with L2 regularization
- **Learning Rate**: Adaptive scheduling with ReduceLROnPlateau
- **Early Stopping**: Patience-based model selection

## 📊 Evaluation Metrics

- **Accuracy**: Overall classification performance
- **Precision**: False positive minimization
- **Recall**: Threat detection sensitivity
- **F1-Score**: Balanced performance metric
- **Confusion Matrix**: Detailed error analysis

## 🤝 Contributing

This project serves as a demonstration of machine learning in cybersecurity. Feel free to:
- Study the implementation
- Suggest improvements
- Adapt for your own use cases

## 📄 License

This project uses the BETH dataset under its respective license terms. The implementation is provided for educational and research purposes.

## 📚 References

- [BETH Dataset Paper](https://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf)
- [Kaggle Dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset)

---

**Note**: Due to file size limitations, the dataset files are excluded from this repository. Download the BETH dataset from Kaggle to run the analysis.

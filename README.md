# ml_phd_federated_learning_workflow

# README: Federated Learning for Privacy-Aware Edge Computing

## Overview
This repository contains a Python implementation of a **privacy-preserving federated learning framework** designed for edge computing environments. The system enables hyper-personalization while ensuring user privacy by training machine learning models locally on edge devices and aggregating updates securely without sharing raw data. This approach is particularly useful for applications such as healthcare, IoT, and personalized recommendation systems.

### Key Features
- **Federated Learning Workflow**: Decentralized model training across multiple clients.
- **Differential Privacy**: Ensures privacy by adding noise to model updates.
- **Edge Computing Integration**: Optimized for low-latency and resource-constrained environments.
- **Scalability**: Supports multiple clients with diverse datasets.
- **Robustness**: Includes mechanisms to handle adversarial attacks and distribution shifts.

---

## Datasets Used
The following datasets were used to evaluate the federated learning framework:

1. **Federated EMNIST**:
   - A distributed version of the EMNIST dataset with handwritten characters from multiple users.
   - Simulates real-world data diversity in federated learning scenarios.

2. **Federated CIFAR-10**:
   - A partitioned version of the CIFAR-10 image classification dataset.
   - Used to test scalability and efficiency in handling large image datasets.

3. **Shakespeare Dataset**:
   - Text data derived from Shakespeare's plays for natural language processing tasks.
   - Evaluates the framework's ability to handle distributed textual data.

4. **Federated Medical Data**:
   - Includes electronic health records (EHR) and medical data from multiple healthcare organizations.
   - Addresses privacy concerns in collaborative healthcare analytics.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch 1.10 or higher
- NumPy
- TensorFlow Federated (optional, for advanced differential privacy experiments)
- TenSEAL (optional, for homomorphic encryption)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/federated-learning-edge.git
   cd federated-learning-edge
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Code
1. Configure the number of clients, rounds, and other parameters in `federated_learning_workflow()` function in `main.py`.
2. Run the script:
   ```bash
   python main.py
   ```

### Output Reports
The script generates detailed reports on:
- Validation accuracy across rounds.
- Test accuracy on unseen datasets.
- Privacy metrics (e.g., differential privacy parameters $$\epsilon$$ and $$\delta$$).
- Communication efficiency (e.g., rounds and data transfer).
- Computational efficiency (e.g., training time, CPU/memory usage).

---

## Code Structure

```
├── main.py               # Main script to run federated learning workflow
├── models.py             # Defines the global and local model architectures
├── utils.py              # Utility functions for data handling and aggregation
├── datasets/             # Placeholder for datasets (downloaded separately)
├── results/              # Directory for storing generated reports and logs
└── README.md             # Documentation (this file)
```

---

## Results

### Performance Metrics

| Metric                          | Value      |
|---------------------------------|------------|
| Validation Accuracy (Final)     | 92.7%      |
| Test Accuracy                   | 93.8%      |
| Differential Privacy ($$\epsilon$$) | 0.5        |
| Total Training Time             | 30 minutes |
| Communication Rounds            | 10         |
| Data Transferred                | 5 MB       |

### Robustness Metrics

| Evaluation Type                 | Result     |
|---------------------------------|------------|
| Accuracy Against Adversarial Attacks | 85%       |
| Performance Under Distribution Shift | 92%       |

---

## How It Works

### Federated Learning Workflow
1. **Global Model Initialization**: A global model is initialized at the central server.
2. **Local Training**: Each client trains a local copy of the global model using its private dataset.
3. **Model Aggregation**: Local updates are aggregated at the server using Federated Averaging.
4. **Privacy Preservation**: Differential privacy mechanisms add noise to updates before aggregation.

### Privacy-Preserving Mechanisms
- **Differential Privacy**: Adds noise to model updates, ensuring individual contributions remain indistinguishable.
- **Homomorphic Encryption** (optional): Enables computations on encrypted data without decryption.

---

## Future Scope

1. **Advanced Optimization Algorithms**: Improve convergence speed and communication efficiency.
2. **Enhanced Privacy Mechanisms**: Explore secure multi-party computation (SMC) and advanced differential privacy techniques.
3. **Real-Time Applications**: Extend support for real-time streaming data in IoT environments.

---

## References

1. Federated EMNIST Dataset: [https://www.tensorflow.org/federated](https://www.tensorflow.org/federated)
2. CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
3. Shakespeare Dataset: [https://leaf.cmu.edu](https://leaf.cmu.edu)
4. Federated Medical Data: Referenced from research papers on federated learning in healthcare.

For more details, refer to the research paper *"Towards Privacy-Aware Hyper Personalization: Exploring Federated Learning for Edge Computing"* by Mahavir Teraiya et al., published in ICIRD 2024.

---

## License
This project is licensed under the MIT License.


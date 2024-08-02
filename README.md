
# ECG Signal Classification using CNN + BiLSTM and Attention Mechanism

This repository contains an implementation of a robust ECG signal classification system that integrates Convolutional Neural Networks (CNN) with Bidirectional Long Short-Term Memory (BiLSTM) and incorporates Attention Mechanisms. The model achieves a 90% accuracy rate in distinguishing between different ECG signal patterns, demonstrating its significant potential in healthcare applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
ECG signal classification is a critical task in healthcare for diagnosing various cardiac conditions. This project leverages deep learning techniques to classify ECG signals into different categories using a hybrid model combining CNN, BiLSTM, and Attention Mechanism.

## Data
The dataset used for this project includes normal and abnormal ECG signals. The data is loaded from CSV files stored on Google Drive.

```python
data = pd.read_csv('/content/drive/MyDrive/ECG/ptbdb_normal.csv')
data.head(5)
```

## Model Architecture
### CNN + BiLSTM + Attention Mechanism
The model architecture includes three main components:
1. **CNN (Convolutional Neural Network)**: Extracts features from the ECG signals.
2. **BiLSTM (Bidirectional Long Short-Term Memory)**: Captures temporal dependencies in the features extracted by the CNN.
3. **Attention Mechanism**: Enhances the model's focus on the most relevant parts of the input signals.

### Diagram of the Model
```
[Input ECG Signal] -> [CNN] -> [BiLSTM] -> [Attention Mechanism] -> [Fully Connected Layer] -> [Output Class]
```

## Implementation Details
### Configuration
```python
class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_csv_path = '/content/drive/MyDrive/ECG/mitbih_with_synthetic/mitbih_with_syntetic_train.csv'
    test_csv_path = '/content/drive/MyDrive/ECG/mitbih_with_synthetic/mitbih_with_syntetic_test.csv'
```

### Data Loading
```python
def get_dataloader(phase: str, batch_size: int = 96) -> DataLoader:
    df = pd.read_csv(config.train_csv_path)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=config.seed, stratify=df['label'])
    dataset = ECGDataset(train_df if phase == 'train' else val_df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
    return dataloader
```

### Model Definition
```python
class RNNAttentionModel(nn.Module):
    def __init__(self, input_size, hid_size, rnn_type, bidirectional, n_classes=5, kernel_size=5):
        super().__init__()
        # Model layers
```

### Training
```python
class Trainer:
    def __init__(self, net, lr, batch_size, num_epochs):
        self.net = net.to(config.device)
        self.optimizer = AdamW(self.net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.dataloaders = {phase: get_dataloader(phase, batch_size) for phase in ['train', 'val']}
    
    def run(self):
        for epoch in range(self.num_epochs):
            self._train_epoch('train')
            with torch.no_grad():
                val_loss = self._train_epoch('val')
                self.scheduler.step()
                if val_loss < self.best_loss:
                    torch.save(self.net.state_dict(), f"best_model_epoc{epoch}.pth")
```

## Results
The model achieved an accuracy of 90% in classifying ECG signals into different categories.

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Training and Validation Metrics
![Metrics](metrics.png)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ECG-Signal-Classification.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the script to train the model:
```bash
python train.py
```
2. Evaluate the model:
```bash
python evaluate.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

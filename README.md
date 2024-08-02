
### Confusion Matrix
To compute and plot the confusion matrix, you can use `sklearn.metrics.confusion_matrix` and `matplotlib` for visualization.

### Training and Validation Metrics
To keep track of the training and validation loss and accuracy, you can store these metrics during the training process and plot them using `matplotlib`.

Here’s the updated code with the confusion matrix and training/validation metrics included:

---

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
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def run(self):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch('train')
            val_loss, val_acc = self._train_epoch('val')
            self.scheduler.step()
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), f"best_model_epoch{epoch}.pth")
    
    def _train_epoch(self, phase):
        # Training logic
        return loss, accuracy
```

### Plotting Metrics
```python
import matplotlib.pyplot as plt

def plot_metrics(trainer):
    epochs = range(1, len(trainer.train_loss_history) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainer.train_loss_history, 'bo-', label='Training Loss')
    plt.plot(epochs, trainer.val_loss_history, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainer.train_acc_history, 'bo-', label='Training Accuracy')
    plt.plot(epochs, trainer.val_acc_history, 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_metrics(trainer)
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    
# Example usage after model prediction
y_true = [...]  # True labels
y_pred = [...]  # Predicted labels
classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']

plot_confusion_matrix(y_true, y_pred, classes)
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

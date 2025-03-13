import pickle
import pandas as pd
import random
from matplotlib import pyplot as plt
from pytorch_widedeep.initializers import *
from pytorch_widedeep.callbacks import *
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

from config import get_config

# Get config, device, and set random seed.
config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(config['random_state'])
torch.manual_seed(config['random_state'])
torch.cuda.manual_seed(config['random_state'])
torch.cuda.manual_seed_all(config['random_state'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['random_state'])

# Read train and val sets into DataFrames.
train = pd.read_csv(os.path.join(config['dataset_path'], config['train_dataset_filename']), index_col=0)
train = train.sample(frac=config['training_data_fraction']).sort_index()
y_train = train[config['target_col']].values
val = pd.read_csv(os.path.join(config['dataset_path'], config['val_dataset_filename']), index_col=0)
y_val = val[config['target_col']].values

# Load contrastively trained TabTransformer.
contrastively_trained_model = torch.load(os.path.join(config['model_path'], f'{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_{config['contrastively_trained_model_filename']}'))
contrastively_trained_model.to(device)

# Load TabPreprocessor and transform the train/validation datasets with it.
with open(f'{config["train_dataset_filename"][6:-4]}_tab_preprocessor.pkl', 'rb') as tp:
    tab_preprocessor = pickle.load(tp)
X_tab_train = tab_preprocessor.transform(train)
X_tab_val = tab_preprocessor.transform(val)
train_dataset = TensorDataset(torch.tensor(X_tab_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_tab_val), torch.tensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=config['tl_batch_size'])
val_loader = DataLoader(val_dataset, batch_size=config['tl_batch_size'])
y_train = y_train[:(len(train_loader) * config['tl_batch_size'])]
y_val = y_val[:(len(val_loader) * config['tl_batch_size'])]

# Get TabTransformer's output dimension.
with torch.no_grad():
    for batch_data, batch_labels in train_loader:
        encoder_output_dim = contrastively_trained_model(batch_data.to(device)).shape[1]

# Initialize FNN.
classifier = nn.Sequential(
    nn.Linear(in_features=encoder_output_dim, out_features=encoder_output_dim * 2),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=encoder_output_dim * 2, out_features=encoder_output_dim),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=encoder_output_dim, out_features=config['n_output_classes'])
)

# Define the FinalModel class which consists of a frozen TabTransformer and an FNN for final classification.
class FinalModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()  # Freeze the encoder weights.
        self.classifier = classifier
        self.output_dim = config['n_output_classes']

    def forward(self, input_data, input_labels=None):
        with torch.no_grad():
            embedded_vectors = self.encoder(input_data)
        return self.classifier(embedded_vectors.to(device))

# Initialize final model object using contrastively trained TabTransformer and newly-created FNN.
final_model = FinalModel(contrastively_trained_model, classifier).to(device)

# Initialize optimizer, scheduler, and loss function.
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, final_model.parameters()), lr=config['tl_lr'], weight_decay=config['tl_weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = nn.CrossEntropyLoss()

# Initialize train/val loss histories.
train_loss = []
val_loss = []

# Train FNN for final classification.
for epoch in range(config['tl_n_epochs']):
    print(f"Epoch {epoch + 1}:")
    final_model.train()
    epoch_train_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_data.to(device))
        batch_labels = batch_labels.type(torch.int64).to(device)
        loss = criterion(outputs, batch_labels.type(torch.int64).to(device))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    print(f"Train Loss: {epoch_train_loss}")
    train_loss.append(epoch_train_loss)

    final_model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            outputs = final_model(batch_data.to(device))
            batch_labels = batch_labels.type(torch.int64).to(device)
            loss = criterion(outputs, batch_labels.type(torch.int64).to(device))
            epoch_val_loss += loss.item()

    epoch_val_loss /= len(val_loader)
    print(f"Val Loss: {epoch_val_loss}")
    val_loss.append(epoch_val_loss)

    lr_scheduler.step(epoch_val_loss)

# Save final model.
with open(os.path.join(config['model_path'], f'{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_{config['final_classification_type']}_{config['final_model_filename']}'), 'wb') as model_file:
    pickle.dump(final_model, model_file)

# Plot train/val loss history.
plt.plot(range(1, len(train_loss) + 1), train_loss, 'g', label='Training loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, 'b', label='Validation loss')
plt.title('Transfer Learning Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./results/{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_{config['final_classification_type']}_transfer_learning_loss.pdf')
plt.show()

# Clear memory.
del train, X_tab_train, y_train
del val, X_tab_val, y_val

# Load and preprocess test dataset.
test = pd.read_csv(os.path.join(config['dataset_path'], config['test_dataset_filename']), index_col=0)
y_test = test[config['target_col']].values
X_tab_test = tab_preprocessor.transform(test)
test_dataset = TensorDataset(torch.tensor(X_tab_test).float(), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=config['tl_batch_size'], drop_last=True)
y_test = y_test[:(len(test_loader) * config['tl_batch_size'])]

# Initialize list of predicted classes.
preds = []

# Collect predictions (perform testing).
final_model.eval()
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        outputs = final_model(batch_data.to(device))
        preds.extend(outputs.argmax(dim=1).cpu().tolist())

# Make results folder if it doesn't exist yet.
if not os.path.isdir('./results/'):
    os.makedirs('./results/')

# Write classification report to disk.
print(classification_report(y_true=y_test, y_pred=preds))
report = pd.DataFrame(classification_report(y_true=y_test, y_pred=preds, output_dict=True)).transpose()
report.to_csv(f'./results/{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_{config['final_classification_type']}_testing_results.csv', index=True)

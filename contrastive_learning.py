import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import torch
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
from sklearn.manifold import TSNE

from config import get_config
import protonet_tools as pnt

# Get config, device, and set random seeds.
config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(config['random_state'])
torch.manual_seed(config['random_state'])
torch.cuda.manual_seed(config['random_state'])
torch.cuda.manual_seed_all(config['random_state'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['random_state'])

# Read train and val sets into DataFrames. Sample a subset of training data.
train = pd.read_csv(os.path.join(config['dataset_path'], config['train_dataset_filename']), index_col=0)
train = train.sample(frac=config['training_data_fraction']).sort_index()
y_train = train[config['target_col']].values
val = pd.read_csv(os.path.join(config['dataset_path'], config['val_dataset_filename']), index_col=0)
y_val = val[config['target_col']].values
test = pd.read_csv(os.path.join(config['dataset_path'], config['test_dataset_filename']), index_col=0)
y_test = test[config['target_col']].values

# Load TabPreprocessor and transform the train/validation datasets with it.
with open(f'{config['train_dataset_filename'][6:-4]}_tab_preprocessor.pkl', 'rb') as tp:
    tab_preprocessor = pickle.load(tp)
X_tab_train = tab_preprocessor.transform(train)
X_tab_val = tab_preprocessor.transform(val)
X_tab_test = tab_preprocessor.transform(test)

# Load self-supervised TabTransformer.
self_supervised_model = torch.load(os.path.join(config['model_path'], f'{config['train_dataset_filename'][6:-4]}_{config['self_supervised_model_filename']}')).model.to(device)

# Initialize train/val loss histories.
train_loss = []
val_loss = []

# Setting the seed
pl.seed_everything(config['random_state'])

classes = np.array(range(0, config['n_output_classes']))
train_set = pnt.dataset_from_labels(X_tab_train, y_train, classes)
val_set = pnt.dataset_from_labels(X_tab_val, y_val, classes)

train_data_loader = data.DataLoader(train_set,
                                    batch_sampler=pnt.FewShotBatchSampler(torch.Tensor(train_set.targets),
                                                                          include_query=True,
                                                                          N_way=config['ct_n_way'],
                                                                          K_shot=config['ct_n_shot'],
                                                                          shuffle=True),
                                    num_workers=4)
val_data_loader = data.DataLoader(val_set,
                                  batch_sampler=pnt.FewShotBatchSampler(torch.Tensor(val_set.targets),
                                                                        include_query=True,
                                                                        N_way=config['ct_n_way'],
                                                                        K_shot=config['ct_n_shot'],
                                                                        shuffle=False,
                                                                        shuffle_once=True),
                                  num_workers=4)

protonet_model, train_loss, val_loss = pnt.train_model(pnt.ProtoNet,
                                                       train_data_loader,
                                                       val_data_loader,
                                                       device,
                                                       model=self_supervised_model,
                                                       lr=config['ct_lr'],
                                                       config=config)

contrastively_trained_model = protonet_model.model
torch.save(contrastively_trained_model, os.path.join(config['model_path'], f'{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_{config['contrastively_trained_model_filename']}'))

# Make results folder if it doesn't exist yet.
if not os.path.isdir('./results/'):
    os.makedirs('./results/')

# Plot train/val loss history.
plt.plot(range(1, len(train_loss) + 1), train_loss, 'g', label='Training loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, 'b', label='Validation loss')
plt.title(f'Contrastive Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.savefig(f'./results/{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_contrastive_training_loss.pdf')
plt.show()

# Define "get_embeddings_in_batches" function that will obtain the embeddings from the TabTransformer in batches.
def get_embeddings_in_batches(model, data_loader):
    embeddings = []
    targets = []
    with torch.no_grad():
        model.eval()
        for batch_data, batch_target in data_loader:
            batch_data = batch_data.to(device)
            batch_embeddings = model(batch_data)
            embeddings.append(batch_embeddings)
            targets.append(batch_target)
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)
    return embeddings, targets

# Transform dataset to prepare it for t-SNE projection.
X_tab_val = pd.DataFrame(X_tab_val)
X_tab_val['target'] = y_val
X_tab_val = X_tab_val.groupby('target').sample(frac=config['tsne_data_percentage'], random_state=config['random_state'])
y_tab_val = X_tab_val['target'].values
X_tab_val = X_tab_val.drop(columns=['target']).values
X_tab_val = torch.tensor(X_tab_val)
y_tab_val = torch.tensor(y_tab_val)
val_dataset = data.TensorDataset(X_tab_val, y_tab_val)
val_loader = data.DataLoader(val_dataset)

# Get TabTransformer's embeddings for t-SNE records.
val_embeddings, val_targets = get_embeddings_in_batches(contrastively_trained_model, val_loader)
tsne_embeddings, tsne_embeddings_labels = get_embeddings_in_batches(contrastively_trained_model, val_loader)

# Perform TSNE projection.
tsne = TSNE(n_components=2, random_state=config['random_state'])
val_tsne = tsne.fit_transform(tsne_embeddings.cpu())
plt.clf()
plt.scatter(val_tsne[:, 0], val_tsne[:, 1], c=tsne_embeddings_labels.cpu(), cmap='rainbow')
plt.title(f'Validation Set t-SNE Projections')
plt.colorbar()
plt.savefig(f'./results/{config['train_dataset_filename'][6:-4]}_{int(config['training_data_fraction'] * 100)}percent_{config['ct_n_shot']}shot_tsne_projections.pdf')
plt.show()

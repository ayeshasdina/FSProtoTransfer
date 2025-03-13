import json
import os
import pickle
import pandas as pd
import torch
import random
from matplotlib import pyplot as plt
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
from pytorch_widedeep.models import FTTransformer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import ContrastiveDenoisingTrainer

from config import get_config

# Get config, device, and set random seeds.
config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(config['random_state'])
torch.manual_seed(config['random_state'])
torch.cuda.manual_seed(config['random_state'])
torch.cuda.manual_seed_all(config['random_state'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Read train and val sets into DataFrames.
train = pd.read_csv(os.path.join(config['dataset_path'], config['train_dataset_filename']), index_col=0)
val = pd.read_csv(os.path.join(config['dataset_path'], config['val_dataset_filename']), index_col=0)

# Set categorical/continuous column names and TabTransformer hidden dimensions.
try:
    cat_embed_cols = config['cat_embed_cols']
    continuous_cols = config['continuous_cols']
    total_features = len(continuous_cols) + len(cat_embed_cols)
except KeyError as e:
    # Adjust accordingly if there are no categorical/continuous columns.
    if 'cat_embed_cols' in str(e):
        cat_embed_cols = None
        continuous_cols = config['continuous_cols']
        total_features = len(continuous_cols)
    else:
        cat_embed_cols = config['cat_embed_cols']
        continuous_cols = None
        total_features = len(cat_embed_cols)

mlp_hidden_dims = [
    total_features * config['input_dim'],
    total_features * config['input_dim'] * 2,
    total_features * config['input_dim']
]

# Initialize TabPreprocessor.
tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols,
    continuous_cols=continuous_cols,
    cols_to_scale='all',
    with_attention=True,
    with_cls_token=True,
)

# Fit TabPreprocessor to train/val data, save TabPreprocessor to file, and use TabPreprocessor to transform val data.
tab_preprocessor.fit(pd.concat([train, val], axis=0))
train = tab_preprocessor.transform(train)
with open(f'{config['train_dataset_filename'][6:-4]}_tab_preprocessor.pkl', 'wb+') as tp:
    pickle.dump(tab_preprocessor, tp)
val = tab_preprocessor.transform(val)

# Set categorical/continuous input to TabTransformer.
cat_embed_input = tab_preprocessor.cat_embed_input if tab_preprocessor.cat_embed_cols is not None else None
continuous_cols = tab_preprocessor.continuous_cols if tab_preprocessor.continuous_cols is not None else None

# Initialize Transformer.
transformer = FTTransformer(
    input_dim=config['input_dim'],
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=cat_embed_input,
    continuous_cols=continuous_cols,
    cat_embed_dropout=config['cat_embed_dropout'],
    cont_embed_dropout=config['cont_embed_dropout'],
    mlp_hidden_dims=mlp_hidden_dims,
    mlp_dropout=config['mlp_dropout'],
)

# Initialize optimizer, scheduler, and projection head dimensions.
optimizer = torch.optim.AdamW(transformer.parameters())
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
projection_head1_dims = [config['input_dim'], config['input_dim'] // 2, config['input_dim'] // 4]
projection_head2_dims = [config['input_dim'], config['input_dim'] // 2, config['input_dim'] // 4]

# Initialize contrastive/denoising trainer.
contrastive_denoising_trainer = ContrastiveDenoisingTrainer(
    model=transformer,
    preprocessor=tab_preprocessor,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    callbacks=[EarlyStopping(patience=config['pt_patience'], restore_best_weights=True), LRHistory(n_epochs=config['pt_n_epochs'])],
    projection_head1_dims=projection_head1_dims,
    projection_head2_dims=projection_head2_dims,
    device=device
)

# Pretrain TabTransformer.
contrastive_denoising_trainer.pretrain(
    X_tab=train,
    X_tab_val=val,
    n_epochs=config['pt_n_epochs'],
    batch_size=config['pt_batch_size']
)

# Save pretrained TabTransformer to file.
contrastive_denoising_trainer.save(path=config['model_path'], model_filename=f'{config['train_dataset_filename'][6:-4]}_{config['self_supervised_model_filename']}')

# Make results folder if it doesn't exist yet.
if not os.path.isdir('./results/'):
    os.makedirs('./results/')

# Load pretraining train/val loss history.
with open(os.path.join(config['model_path'], 'history/train_eval_history.json'), 'r') as file:
    loss_history = json.load(file)

# Plot train/val loss history.
train_loss_history = loss_history['train_loss']
val_loss_history = loss_history['val_loss']
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'g', label='Training loss')
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'b', label='Validation loss')
plt.title('Self-Supervised Learning Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./results/{config['train_dataset_filename'][6:-4]}_self_supervised_learning_loss.pdf')
plt.show()

# Plot learning rate (LR) history.
lr_history = contrastive_denoising_trainer.lr_history['lr_0']
plt.clf()
plt.plot(range(1, len(lr_history) + 1), lr_history, 'r', label='Learning Rate')
plt.title(f'Self-Supervised Learning LR History')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.savefig(f'./results/{config['train_dataset_filename'][6:-4]}_self_supervised_learning_lr_history.pdf')
plt.show()

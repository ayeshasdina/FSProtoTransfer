def get_config():
    return {
        # Paths and Filenames
        'dataset_path': './datasets',  # Folder to store datasets.
        'model_path': './models/',  # Folder to store trained models.
        'self_supervised_model_filename': 'self_supervised_model.pt',
        'contrastively_trained_model_filename': 'contrastively_trained_model.pt',
        'final_model_filename': 'final_model.pkl',

        'random_state': 42,  # Random seed to use throughout training (for reproducibility).

        # Dataset Parameters
        # Car-Hacking
        'train_dataset_filename': 'train_car_hacking.csv',
        'val_dataset_filename': 'val_car_hacking.csv',
        'test_dataset_filename': 'test_car_hacking.csv',
        'cat_embed_cols': ['ID', 'Data0', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7'],  # Categorical column names in the dataset.
        'continuous_cols': ['Timestamp'],  # Numerical column names in the dataset.
        'target_col': 'Label',  # Name of the output label column in the dataset.
        'n_output_classes': 5,
        # University of Minho
        # 'train_dataset_filename': 'train_university_of_minho.csv',
        # 'val_dataset_filename': 'val_university_of_minho.csv',
        # 'test_dataset_filename': 'test_university_of_minho.csv',
        # 'cat_embed_cols': ['longAcceleration', 'bitLen', 'diffAcc'],
        # 'continuous_cols': ['diffTime', 'heading', 'speed', 'latitude', 'longitude', 'diffPos', 'diffSpeed', 'diffHeading'],
        # 'target_col': 'isAttack',
        # 'n_output_classes': 5,

        # FTTransformer Parameters
        'cat_embed_dropout': 0.2,  # Categorical embeddings dropout. [0.2]
        'cont_embed_dropout': 0.2,  # Continuous embeddings dropout. [0.2]
        'input_dim': 64,  # Number of embeddings used to encode categorical and/or continuous features. [64]
        'mlp_dropout': 0.2,  # Final MLP dropout. [0.2]

        # Self-Supervised Pretraining Parameters
        'pt_n_epochs': 100,  # Number of self-supervised learning epochs. [100]
        'pt_batch_size': 500,  # Self-supervised learning batch size. [500]
        'pt_patience': 50,  # Patience (number of epochs to wait since last decrease in val loss) during self-supervised pretraining. [50]

        # Training Data Parameters
        'training_data_fraction': 0.2,  # Percentage of training data to use for contrastive and final classification. [0.2]

        # Few-Shot Learning and Contrastive Training Parameters
        'ct_n_epochs': 200,  # Number of epochs for contrastive training. [200]
        'ct_lr': 2e-4,  # Learning rate to use in contrastive trianing. [2e-4]
        'ct_n_way': 5,  # Number of classes per episode in contrastive training. [5]
        'ct_n_shot': 10,  # Number of support examples per class in prototypical network training. [10]
        'tsne_data_percentage': 0.05,  # Percentage of each class in the validation set to use in t-SNE projection. [0.05]

        # Transfer Learning Parameters
        'final_classification_type': 'fnn',  # Final classification model type. ['fnn']
        'tl_n_epochs': 100,  # Number of epochs for transfer learning. [100]
        'tl_batch_size': 1000,  # Batch size to use in transfer learning. [1000]
        'tl_lr': 0.001,  # Learning rate for transfer learning. [0.001]
        'tl_weight_decay': 1e-6,  # Weight decay for transfer learning. [1e-6]
    }


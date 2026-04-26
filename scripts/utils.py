class Config:
    """Конфигурации проекта."""
    RANDOM_SEED = 42

    TEXT_MODEL_NAME = 'bert-base-uncased'
    IMAGE_MODEL_NAME = 'tf_efficientnet_b0'

    TEXT_MODEL_UNFREEZE = 'encoder.layer.11|pooler' 
    IMAGE_MODEL_UNFREEZE = 'blocks.6|conv_head|bn2'

    VAL_SIZE = 0.15
    BATCH_SIZE = 16
    TEXT_LR = 2e-5
    IMAGE_LR = 3e-5
    CLASSIFIER_LR = 5e-4
    
    EPOCHS = 30
    DROPOUT = 0.2
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1 
    
    DATA_DIR = './data'
    IMAGE_DIR = './data/images'
    SAVE_MODEL_DIR = './model'
    SAVE_MODEL_NAME = 'best_model.pth'

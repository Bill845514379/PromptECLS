cfg = {
    'gpu_id': 0,
    'max_len': 220,
    'train_batch_size': 4,
    'test_batch_size': 32,
    'learning_rate': 1e-5,
    'epoch': 10,
    'K': 16,
    'Kt': 1000,
    # 'template': '[X1] [X2]? [MASK].',
    'template': '[X1] ? [MASK] , [X2]',
    'answer': ['No', 'Yes'],
    'device': 'TPU',
    'optimizer': 'Adam',
    'word_size': 50265
}

hyper_roberta = {
    'word_dim': 1024,
    'dropout': 0.1
}

path = {
    'neg_path': 'data/rt-polaritydata/neg_label.txt',
    'pos_path': 'data/rt-polaritydata/pos_label.txt',
    'roberta_path': 'roberta-large'
}

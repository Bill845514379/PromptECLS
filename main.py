import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from common.text2id import X_data2id, get_answer_id
import os
import torch
from config.cfg import cfg, path, hyper_roberta
from common.load_data import load_data, tokenizer, data_split, generate_template
from model.PromptMask import PromptMask
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.autograd import Variable
from common.metric import ScorePRF
from common.set_random_seed import setup_seed
import pytorch_lightning as pl
import time

print('data preprocessing ...')
pos_X, pos_y = load_data(path['pos_path'])
train_pos_X, train_pos_y, test_pos_X, test_pos_y = data_split(pos_X, pos_y, cfg['K'], cfg['Kt'])
neg_X, neg_y = load_data(path['neg_path'])
train_neg_X, train_neg_y, test_neg_X, test_neg_y = data_split(neg_X, neg_y, cfg['K'], cfg['Kt'])

train_X0 = np.hstack([train_pos_X, train_neg_X])
train_y0 = np.hstack([train_pos_y, train_neg_y])
test_X0 = np.hstack([test_pos_X, test_neg_X])
test_y0 = np.hstack([test_pos_y, test_neg_y])

train_X, train_y = generate_template(train_X0, train_X0, train_y0, train_y0)
test_X, test_y = generate_template(test_X0, train_X0, test_y0, train_y0)

train_X, test_X = X_data2id(train_X, tokenizer), X_data2id(test_X, tokenizer)
train_y, answer_map = get_answer_id(train_y, tokenizer)
test_y, _ = get_answer_id(test_y, tokenizer)

train_X, train_y = torch.tensor(train_X), torch.tensor(train_y)
test_X, test_y = torch.tensor(test_X), torch.tensor(test_y)

train_data = TensorDataset(train_X, train_y)
test_data = TensorDataset(test_X, test_y)

loader_train = DataLoader(
    dataset=train_data,
    batch_size=cfg['train_batch_size'],
    shuffle=True,
    num_workers=4,
    drop_last=False
)

loader_test = DataLoader(
    dataset=test_data,
    batch_size=cfg['K'] * 2,
    shuffle=False,
    num_workers=4,
    drop_last=False
)

print('start training ... ')
net = PromptMask(answer_map)

trainer = pl.Trainer(tpu_cores=8, max_epochs=10)
trainer.fit(net, loader_train)
print('start testing ... ')
trainer.test(net, loader_test)
print(net.acc)

# Base Package
import os
import time
import copy
import datetime
from pprint import pformat
import logging
import numpy as np
import pandas as pd 
from tqdm import tqdm_notebook
import tqdm
import json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

# Our Package
from config import ParameterSetting
from dataset import SoundDataset, get_melspec
from models import ModelEMA 
from metrics import roc_auc 
from ops import AdamW

logger = logging.getLogger(__file__)

def main():
    parser = ArgumentParser()
    # data or model path setting
    parser.add_argument("--csv_path", type=str, default='/DATA/meta_train.csv', help='the path of train csv file')
    parser.add_argument("--data_dir", type=str, default="/DATA/train", help="the directory of sound data")
    # training parameter setting
    parser.add_argument("--model_name", type=str, default='EFFb0', choices=['EFFb0'], help='the algorithm we used')
    parser.add_argument("--val_split", type=float, default=0.1, help="the ratio of validation set. 0 means there's no validation dataset")
    parser.add_argument("--epochs", type=int, default=20, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_class", type=int, default=6, help="number of classes")
    # data augmentation setting
    parser.add_argument("--spec_aug", action='store_true', default=False)
    parser.add_argument("--time_drop_width", type=int, default=64)
    parser.add_argument("--time_stripes_num", type=int, default=2)
    parser.add_argument("--freq_drop_width", type=int, default=8)
    parser.add_argument("--freq_stripes_num", type=int, default=2)
    # proprocessing setting
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--nfft", type=int, default=200)
    parser.add_argument("--hop", type=int, default=80)
    parser.add_argument("--mel", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    ##################
    # config setting #
    ##################

    # 參數設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_ema = True

    save_model_dir = 'eff_0710'
    if not os.path.isdir(f'./model/{save_model_dir}'):
        os.mkdir(f'./model/{save_model_dir}')

    train_record = {}
    for i in ['accuracy', 'loss', 'auc']:
        for j in range(5):
            train_record[f'valid_{i}_{j}'] = []

    # if not os.path.exists('params.save_root'):
    #     os.mkdir(params.save_root)
    #     print("create folder: {}".format(params.save_root))
    #     if not os.path.exists(os.path.join(params.save_root, 'snapshots')):
    #         os.mkdir(os.path.join(params.save_root, 'snapshots'))
    #     if not os.path.exists(os.path.join(params.save_root, 'log')):
    #         os.mkdir(os.path.join(params.save_root, 'log'))

    ####################################
    # data preparing & model training  #
    ####################################

    # 訓練資料集
    train_sepc = []
    for file_name in tqdm.notebook.tqdm(meta_train.Filename):
        train_sepc.append(get_melspec('train',file_name))
    train_sepc   

    # 5-fold 
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    for fold, (train_index, val_index) in enumerate(skf.split(train_sepc, meta_train.Label.values)):
        print(f'---- fold{fold} --------------------------------------')

        print("Preparing training/validation data...")
        trn_dataset = SoundDataset(np.array(train_sepc)[train_index], meta_train.Label.values[train_index])
        val_dataset = SoundDataset(np.array(train_sepc)[val_index], meta_train.Label.values[val_index], mode='val')
        # num_workers：使用多進程加載的進程數、pin_memory：是否將數據保存在pin memory區，pin memory中的數據轉到GPU會快一些
        trn_loader = DataLoader(trn_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        print("Preparing efficientnet model...")
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6,in_channels=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)    

        best_loss = 10; best_acc = 0.0; patience = 0; patience_total = 150; best_auc = 0.5    
        model_path = f'./model/{save_model_dir}/model_loss_fold_{fold}.ckpt'
        model_path_2 = f'./model/{save_model_dir}/model_acc_fold_{fold}.ckpt'
        model_path_3 = f'./model/{save_model_dir}/model_auc_fold_{fold}.ckpt'
        
        if use_ema:
            ema_model = ModelEMA(model, decay=0.99)

        # ---- train -------------------------------------------------
        for epoch in range(0,1000):

            # ---------- Training ----------
            model.train()
            train_loss = 0; reg_loss = 0; correct = 0; total = 0            

            # Iterate the training set by batches.
            # for batch in tqdm.notebook.tqdm(trn_loader):
            for batch in trn_loader:
            
                # ---------- mixup -----------
                inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs = inputs.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if use_ema:
                    ema_model.update(model)

        # ---------- Validation ----------
        if use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        test_model.eval()

        # Validation 紀錄
        valid_loss = []; valid_accs = []; val_label = []; val_pred = []; val_pred_p = []
        # Iterate the validation set by batches.
        for batch in val_loader:

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # 驗證階段不用梯度的資訊，取消梯度以加速運算
            with torch.no_grad():
                logits = test_model(imgs.to(device).float())

            # 計算每批 loss 跟 accuracy
            loss = criterion(logits, labels.to(device))   
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean() 
            val_pred.append(logits.argmax(dim=-1).to('cpu').numpy())
            val_label.append(labels.numpy())
            val_pred_p.append(nn.Softmax(dim=-1)(logits[:,:6]).to('cpu').numpy())
            valid_loss.append(loss.item())
            valid_accs.append(acc.item())

        # 計算整體 loss 跟 accuracy
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        
        cm = pd.crosstab(np.hstack(val_label),np.hstack(val_pred))
        # print(cm)roc_auc_score(y_true, y_score, multi_class=multi_class, average=average)
        valid_auc = roc_auc(np.eye(6)[list(np.hstack(val_label))],np.vstack(val_pred_p))

        train_record[f'valid_loss_{fold}'].append(valid_loss)
        train_record[f'valid_accuracy_{fold}'].append(valid_acc)
        train_record[f'valid_auc_{fold}'].append(valid_auc)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(test_model.state_dict(), model_path)
            print(f'epoch_{epoch} ','saving model with loss {:.5f}'.format(valid_loss))
            patience=0
        else:
            patience+=1
            if patience>patience_total:
                print('loss')
                break
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(test_model.state_dict(), model_path_2)
            print(f'epoch_{epoch} ','saving model with accuracy {:.5f}'.format(valid_acc))
            patience=0
        else:
            patience+=1
            if patience>patience_total:
                print('acc')
                break  
        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(test_model.state_dict(), model_path_3)
            print(f'epoch_{epoch} ','saving model with auc {:.5f}'.format(valid_auc))
            patience=0
        else:
            patience+=1
            if patience>patience_total:
                print('auc')
                break
    
    # Save valid score
    filename = "valid_score.json"
    with open(filename, "w") as file: 
        json.dump(train_record, file) 

    ###################
    # training ending #
    ###################

if __name__ == '__main__':
    main()

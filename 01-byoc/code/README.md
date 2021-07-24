# Efficientnetb0 for Sound Classification

This repo contains a final model of sound classfication - EFFb0 structure trained on our own competition dataset.

* [Environments](#environments)
* [Label List](#label-list) 
* [Train EFFb0](#train-effb0-model)
  * [Prepare train/test meta data](#prepare-train/test-meta-data)
  * [Train the model](#train-the-model)
* [Saved results](#saved-results)
* [Visualize the results](#visualize-the-results)


## Environments

The codebase is developed with Python 3.7.10. Install requirements as follows:

```bash
pip install -r requirements_colab.txt
```
## Label List

到時候比賽會有10類~~~

| Label Number | Class Name |
|----------|-------|
| 0 | Barking  |
| 1 | Howling  |
| 2 | Crying  |
| 3 | COSmoke  |
| 4 | GlassBreaking  |
| 5 | Other  |

## Train EFFb0 Model

The training process consists of two parts: 
  1. Prepare train/test meta csv. 
  2. Train the model

### Prepare train/test meta data

Users need to prepare train meta csv data and follow this type of format:

| Filename | Label | Remark |
|----------|-------|--------|
| train_00001 | 0  | Barking |
| train_00002 | 0  | Barking |
| train_00601 | 3  | COSmoke |
| train_00801 | 4  | GlassBreaking |

需指定 --data_dir --csv_path 對應的資料夾路徑

When users train the model, they need to specify ```--data_dir```, which is the root directory of sound data. ```--csv_path```, which is metadata of training. Then, the dataset will load wav data and label from this csv data.

Users need to prepare test meta csv data and follow this type of format:

| Filename | Barking | Howling | Crying | COSmoke | GlassBreaking | Other |
|----------|-------|--------|-------|--------|-------|--------|
| public_00006 |0|1|0|0|0|0|
| public_00009 |0|0|1|0|0|0|
| public_00010 |0|0|0|0|0|1|
| public_00030 |0|0|0|1|0|0|

The test meta csv contains ground truth of testing sound data. 

### Train the model

Users can train EFFb0 model by executing the following commands.

```bash
# AWS的參數設定
python train.py --csv_path=./meta_train.csv --data_dir=./train --epochs=50 --val_split 0.1 --preload
# 我們自己的設定
python train.py --csv_path=./meta_train.csv --data_dir=./train --epochs=50
```

```--val_split``` - the ratio of validation size split from training data

```--preload``` - whether to convert wav data to melspectrogram first before start training

The interface will be printed on screen like this:

```
Epoch 3/50
----------
8/9 [======>.] - ETA: 0s - train loss in batch: 1.2447 - train acc in batch: 0.7868
9/9 [========] - 7s 730ms/step - train loss in batch: 1.2447 - train acc in batch: 0.7868 - train epoch loss: 1.2528 - train acc: 0.7741 - train precision: 0.7805 - train recall: 0.7742 - train f1: 0.7730
0/1 [........] - ETA: 0s - val loss in batch: 0.0000e+00 - val acc in batch: 0.0000e+00
finish this epoch in 0m 0s
1/1 [========] - 0s 155ms/step - val loss in batch: 0.0000e+00 - val acc in batch: 0.0000e+00 - val epoch loss: 1.8093 - val acc: 0.2083 - val precision: 0.0347 - val recall: 0.1667 - val f1: 0.0575
```

## Saved results

The checkpoints will be saved in ```results/snapshots/[model_name]```.
The log information will be saved in ```results/log/[model_name]```.


```bash
root
├── results
│    ├── snapshots
│    |    └── model_name
│    |       └── epoch_001_valloss ... .pkl
│    |
│    └── log
│         └── model_name
│            ├── cfm.png
│            ├── events.out.tfevents.1599822797.tomorun.14975.0
│            └── classification_report.txt
│
├── losses.py
├── ops.py
├── models.py
├── config.py
├── train.py
├── test.py
├── dataset.py
├── README.md
├── utils.py
└── metrics.py
```
### Our Training Code Architecture
```bash
root
├── model_efn
│    └── eff_class10_0714
│        └── model_acc_fold_ ... .ckpt
│
├── README.md
├── losses.py
├── ops.py
├── models.py
├── metrics.py
├── dataset.py
├── train.py
└── utils.py
```

## Visualize the results

Running 查看訓練結果

```bash
tensorboard --logdir=results
```

from the command line and then navigating to [http://localhost:6006](http://localhost:6006)

## Test the model

Users can test the model with the following example command:

```bash
# AWS的寫法
python test.py --test_csv ./meta_public_test.csv --data_dir ./private_test --model_name VGGish --model_path [path_of_models] --saved_root results/test --saved_name test_result
# 我們的寫法
直接在jupyter寫啦
```

The testing results will be saved in ```--saved_root``` and the prefix of files will be ```--saved_name```. You'll get the classfication report and the confusion matrix in txt format.
# M3AE

This is the official implementation of [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training]() at MICCAI-2022.

## Table of Contents
- [Requirements](#requirements)
- [Download](#download-m3ae)
- [Pre-training](#pre-training)
- [Downstream Evaluation](#downstream-evaluation)

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Download M3AE
You can download the models we pre-trained and fine-tuned in the corresponding datasets from [here](https://drive.google.com/drive/folders/1lS2fm8RqFxoKQ8cgYD9hMFoOuHuKZNoY?usp=sharing).

## Pre-training
### 1. Dataset Preparation
Please organize the pre-training datasets as the following structure:
```angular2
root:[data]
+--pretrain_data
| +--roco
| | +--val
| | +--test
| | +--train
| +--medicat
| | +--release
| | +--net
```


### 2. Pre-processing
Run the following command to pre-process the data:
```angular2
python prepro/prepro_pretraining_data.py
```
to get the following arrow files:
```angular2
root:[data]
+--pretrain_arrows
| +--medicat_train.arrow
| +--medicat_val.arrow
| +--medicat_test.arrow
| +--roco_train.arrow
| +--roco_val.arrow
| +--roco_test.arrow
```


### 3. Pre-training
Now we can start to pre-train the m3ae model:
```angular2
bash run_scripts/pretrain_m3ae.sh
```

## Downstream Evaluation
### 1. Dataset Preparation
Please organize the fine-tuning datasets as the following structure:
```angular2
root:[data]
+--finetune_data
| +--melinda
| | +--train.csv
| | +--dev.csv
| | +--test.csv
| | +--melinda_images
| +--slack
| | +--train.json
| | +--validate.json
| | +--test.json
| | +--imgs
| +--vqa_rad
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
| +--medvqa_2019
| | +--val
| | +--test
| | +--train
```

### 2. Pre-processing
Run the following command to pre-process the data:
```angular2
python prepro/prepro_finetuning_data.py
```
to get the following arrow files:
```angular2
root:[data]
+--finetune_arrows
| +--vqa_vqa_rad_train.arrow
| +--vqa_vqa_rad_val.arrow
| +--vqa_vqa_rad_test.arrow
| +--vqa_slack_train.arrow
| +--vqa_slack_test.arrow
| +--vqa_slack_val.arrow
| +--vqa_medvqa_2019_train.arrow
| +--vqa_medvqa_2019_val.arrow
| +--vqa_medvqa_2019_test.arrow
| +--cls_melinda_train.arrow
| +--cls_melinda_val.arrow
| +--cls_melinda_test.arrow
| +--irtr_roco_train.arrow
| +--irtr_roco_val.arrow
| +--irtr_roco_test.arrow
```

### 3. Fine-Tuning
Now you can start to fine-tune the m3ae model:
```angular2
bash run_scripts/finetune_m3ae.sh
```

### 4. Test
You can also test our fine-tuned models directly:
```angular2
bash run_scripts/test_m3ae.sh
```
NOTE: This is a good way to check whether your environment is set up in the same way as ours (if you can reproduce the same results).

## Acknowledgement
The code is based on [ViLT](https://github.com/dandelin/ViLT), [METER](https://github.com/zdou0830/METER) and [MAE](https://github.com/facebookresearch/mae).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations
If M3AE is useful for your research, please consider citing:
```angular2
@inproceedings{chen2022m3ae,
  title={Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training},
  author={Chen, Zhihong and Du, Yuhao and Hu, Jinpeng and Liu, Yang and Li, Guanbin and Wan, Xiang and Chang, Tsung-Hui},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022},
  organization={Springer}
}
```

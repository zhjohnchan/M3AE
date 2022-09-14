num_gpus=1
per_gpu_batchsize=16

# === VQA ===
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/pretrained/m3ae.ckpt

python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/pretrained/m3ae.ckpt \
 clip_resizedcrop

python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_medvqa_2019 \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/pretrained/m3ae.ckpt \
 clip_resizedcrop

# === CLS ===
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_cls_melinda_p_meth_label \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/pretrained/m3ae.ckpt \
 clip_resizedcrop

# === IRTR ===
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_irtr_roco get_recall_metric=True \
 pwdper_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=288 \
 tokenizer=downloaded/roberta-base \
 test_only=True \
 load_path=downloaded/pretrained/m3ae.ckpt

python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_irtr_roco get_recall_metric=False \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/pretrained/m3ae.ckpt \
 clip_resizedcrop
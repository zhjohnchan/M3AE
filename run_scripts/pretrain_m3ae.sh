num_gpus=8
per_gpu_batchsize=32

python main.py \
 with data_root=data/pretrain_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_pretrain_m3ae \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=288 max_text_len=64 \
 tokenizer=downloaded/roberta-base

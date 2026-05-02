@echo off
cd /d E:\学习\molfcddi
D:\Users\Li\miniconda3\envs\pytorch\python.exe experiment_ddi.py --split_name S1_random --mode multiclass --split_dir "E:\学习\molfcddi\data\splits" --gpu 0 --epochs 50 --batch_size 256 --init_lr 0.0001 --max_lr 0.001 --final_lr 0.0001 --warmup_epochs 5 --hidden_size 300 --depth 3 --ffn_num_layers 2 --dropout 0.1 --seed 42 > logs\S1_random_multiclass.log 2>&1

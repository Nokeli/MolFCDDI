@echo off
cd /d E:\学习\molfcddi
D:\Users\Li\miniconda3\envs\pytorch\python.exe ddi_train.py --split_name S2_one_unseen --mode binary --split_dir "E:\学习\molfcddi\data\splits" --gpu 0 --epochs 50 --batch_size 256 --seed 42 >> logs/exp_S2_bin.log 2>&1

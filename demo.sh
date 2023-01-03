#!/bin/bash
#Train x2 PAEDSR_baseline
python main.py --n_GPUs 4 --dir_data ../../ --rgb_range 1 --save_models --lr 1e-4 --decay 200-400-600-800 --test_every 1000 --chop --save_results --n_resblocks 16 --n_feats 64 --res_scale 1 --batch_size 4 --model PAEDSR --scale 2 --patch_size 96 --save EDSR_PA_baseline_x2 --data_train DIV2K


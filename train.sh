#!/usr/bin/env bash
MVS_TRAINING="/mnt/sgvrnas/youngju/DTU"
python train.py --dataset=dtu_yao --batch_size=4 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/train_mvs $@

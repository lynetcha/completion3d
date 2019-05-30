#!/usr/bin/env bash

# Data Parameters
DATASET='shapenet'
NPTS=$((1*(2048)))

# Model Parameters
NET='AtlasNet'
CODE_NFTS=1024
DIST_FUN='chamfer'
NB_PRIMITIVES=4

# Training Parameters
TRAIN=1
EVAL=$((1-$TRAIN))
RESUME=0
BENCHMARK=0
OPTIM='adagrad'
LR=1e-2
EPOCHS=300
SAVE_EPOCH=1
TEST_EPOCH=$SAVE_EPOCH
BATCH_SIZE=32
NWORKERS=4

# Data Augmentation
SCALE=0
ROT=1
MIRROR=0.5

PROGRAM="main.py"

python -u $PROGRAM --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE \
    --nworkers  $NWORKERS --NET $NET --dataset $DATASET \
    --pc_augm_scale $SCALE --pc_augm_rot $ROT --pc_augm_mirror_prob $MIRROR \
    --eval $EVAL --optim $OPTIM --code_nfts $CODE_NFTS  --benchmark $BENCHMARK \
    --resume $RESUME --npts $NPTS  --dist_fun $DIST_FUN \

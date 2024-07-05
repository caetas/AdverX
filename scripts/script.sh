#!/bin/bash

cd ../src/adverx
python train.py --hidden_dims 64 128 256 512 512 --latent_dim 1024 --n_epochs 100 --batch_size 128 --lr 3e-4
poweroff
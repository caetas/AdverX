#!/bin/bash

cd ../src/adverx
python train_DCGAN.py --latent_dim 1024 --n_epochs 100 --batch_size 128 --in_machine siemens
python train_DCGAN.py --latent_dim 1024 --n_epochs 100 --batch_size 128 --in_machine philips
python train_DCGAN.py --latent_dim 1024 --n_epochs 100 --batch_size 128 --in_machine konica
python train_DCGAN.py --latent_dim 1024 --n_epochs 100 --batch_size 128 --in_machine ge
python train_DCGAN.py --latent_dim 1024 --n_epochs 100 --batch_size 128 --in_machine gmm
poweroff
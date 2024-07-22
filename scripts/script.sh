#!/bin/bash

cd ../src/adverx
#python train_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine siemens
python train_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine philips
python train_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine konica
python train_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine ge
python train_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine gmm
#python train_PresGAN.py --nz 1024 --n_epochs 100 --batch_size 128 --in_machine siemens --ngf 64 --ndf 64 --lambda_ 0.00025 --sigma_min 0.001 --sigma_max 0.3 --restrict_sigma 1
#python train_PresGAN.py --nz 1024 --n_epochs 100 --batch_size 128 --in_machine philips --ngf 64 --ndf 64 --lambda_ 0.00025 --sigma_min 0.001 --sigma_max 0.3 --restrict_sigma 1
#python train_PresGAN.py --nz 1024 --n_epochs 100 --batch_size 128 --in_machine konica --ngf 64 --ndf 64 --lambda_ 0.00025 --sigma_min 0.001 --sigma_max 0.3 --restrict_sigma 1
#python train_PresGAN.py --nz 1024 --n_epochs 100 --batch_size 128 --in_machine ge --ngf 64 --ndf 64 --lambda_ 0.00025 --sigma_min 0.001 --sigma_max 0.3 --restrict_sigma 1
#python train_PresGAN.py --nz 1024 --n_epochs 100 --batch_size 128 --in_machine gmm --ngf 64 --ndf 64 --lambda_ 0.00025 --sigma_min 0.001 --sigma_max 0.3 --restrict_sigma 1
poweroff
#!/bin/bash

cd ../src/adverx
#python eval_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine siemens
#python eval_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine philips
#python eval_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine konica
#python eval_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine ge
#python eval_GLOW.py --L 4 --hidden_channels 128 --batch_size 100 --n_epochs 100 --in_machine gmm
python eval_VAE.py --hidden_dims 64 128 256 512 512 --latent_dim 1024 --n_epochs 200 --batch_size 128 --lr 3e-4 --sample_and_save_freq 10 --in_machine siemens --patches_image 32 --checkpoint ./../../models/VAE/VAE_siemens.pt
python eval_VAE.py --hidden_dims 64 128 256 512 512 --latent_dim 1024 --n_epochs 200 --batch_size 128 --lr 3e-4 --sample_and_save_freq 10 --in_machine philips --patches_image 40 --checkpoint ./../../models/VAE/VAE_philips.pt
python eval_VAE.py --hidden_dims 64 128 256 512 512 --latent_dim 1024 --n_epochs 200 --batch_size 128 --lr 3e-4 --sample_and_save_freq 10 --in_machine konica --patches_image 36 --checkpoint ./../../models/VAE/VAE_konica.pt
python eval_VAE.py --hidden_dims 64 128 256 512 512 --latent_dim 1024 --n_epochs 200 --batch_size 128 --lr 3e-4 --sample_and_save_freq 10 --in_machine ge --patches_image 42 --checkpoint ./../../models/VAE/VAE_ge.pt
python eval_VAE.py --hidden_dims 64 128 256 512 512 --latent_dim 1024 --n_epochs 200 --batch_size 128 --lr 3e-4 --sample_and_save_freq 10 --in_machine gmm --patches_image 42 --checkpoint ./../../models/VAE/VAE_gmm.pt
#poweroff

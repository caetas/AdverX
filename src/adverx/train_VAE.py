from data.Dataloader import train_loader
from models.VAE import VAE
from utils.util import parse_args_VAE
import wandb

args = parse_args_VAE()
wandb.init(project='VAE',
           config={
               'batch_size': args.batch_size,
               'n_epochs': args.n_epochs,
               'lr': args.lr,
               'latent_dim': args.latent_dim,
               'hidden_dims': args.hidden_dims,
               'sample_and_save_freq': args.sample_and_save_freq,
               'kld_weight': args.kld_weight,
               'loss_type': args.loss_type,
               'patches_image': args.patches_image,
               'split': args.split,
               'in_machine': args.in_machine  
           },
           name=f"VAE_{args.in_machine}")

loader = train_loader(args.batch_size, args.patches_image, args.split, args.in_machine)
model = VAE(128,1,args)
model.train_model(loader, args.n_epochs)
wandb.finish()
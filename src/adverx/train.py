from data.Dataloader import train_loader
from models.AdverX import AdverX
from utils.util import parse_args_AdverX
import wandb

args = parse_args_AdverX()
wandb.init(project='AdverX',
           config={
               'batch_size': args.batch_size,
               'n_epochs': args.n_epochs,
               'lr': args.lr,
               'latent_dim': args.latent_dim,
               'hidden_dims': args.hidden_dims,
               'num_samples': args.num_samples,
               'gen_weight': args.gen_weight,
               'recon_weight': args.recon_weight,
               'sample_and_save_frequency': args.sample_and_save_frequency,
               'kld_weight': args.kld_weight,
               'loss_type': args.loss_type,
               'patches_image': args.patches_image,
               'split': args.split  
           })
loader = train_loader(args.batch_size, args.patches_image, args.split)
model = AdverX(128,1,args)
model.train_model(loader)
wandb.finish()
from data.Dataloader import train_loader
from models.DCGAN import DCGAN
from utils.util import parse_args_DCGAN
import wandb

args = parse_args_DCGAN()

wandb.init(project='DCGAN',
               config={
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'latent_dim': args.latent_dim,
                   'd': args.d,
                   'lrg': args.lrg,
                   'lrd': args.lrd,
                   'beta1': args.beta1,
                   'beta2': args.beta2,
                   'sample_and_save_freq': args.sample_and_save_freq,
                   'patches_image': args.patches_image,
                   'split': args.split,
                   'in_machine': args.in_machine 
               },
               name = 'DCGAN_{}'.format(args.in_machine))

loader = train_loader(args.batch_size, args.patches_image, args.split, args.in_machine)
model = DCGAN(args, 1, 128)
model.train_model(loader)
from data.Dataloader import train_loader
from models.PrescribedGAN import PresGAN
from utils.util import parse_args_PresGAN
import wandb

args = parse_args_PresGAN()

wandb.init(project='PresGAN',
               config={
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'nz': args.nz,
                     'ngf': args.ngf,
                        'ndf': args.ndf,
                        'lrD': args.lrD,
                        'lrG': args.lrG,
                        'in_machine': args.in_machine,
                        'in_patches': args.in_patches,
                        'out_patches': args.out_patches,
                        'split': args.split,
                        'patches_image': args.patches_image
               },
               name = 'PresGAN_{}'.format(args.in_machine))

loader = train_loader(args.batch_size, args.patches_image, args.split, args.in_machine)
model = PresGAN(128, 1, args)
model.train_model(loader)
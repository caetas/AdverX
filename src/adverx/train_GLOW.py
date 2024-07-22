from data.Dataloader import train_loader
from models.Glow import Glow
from utils.util import parse_args_Glow
import wandb

args = parse_args_Glow()

wandb.init(project='GLOW',
               
               config={
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "n_epochs": args.n_epochs,
                    "in_machine": args.in_machine,
                    "hidden_channels": args.hidden_channels,
                    "K": args.K,
                    "L": args.L,
                    "actnorm_scale": args.actnorm_scale,
                    "flow_permutation": args.flow_permutation,
                    "flow_coupling": args.flow_coupling,
                    "LU_decomposed": args.LU_decomposed,
                    "learn_top": args.learn_top,
                    "y_condition": args.y_condition,
                    "num_classes": args.num_classes,
                    "n_bits": args.n_bits,  
               },

                name = 'GLOW_{}'.format(args.in_machine))
    
loader = train_loader(args.batch_size, args.patches_image, args.split, args.in_machine, normalize = False)
model = Glow(image_shape        =   (128,128,1), hidden_channels    =   args.hidden_channels, args=args)
model.train_model(loader, args)
from data.Dataloader import test_loader
from models.AdverX import AdverX
from utils.util import parse_args_AdverX
import torch

args = parse_args_AdverX()
model = AdverX(128,1,args)
if args.checkpoint is not None:
    model.vae.load_state_dict(torch.load(args.checkpoint))
if args.discriminator_checkpoint is not None:
    model.discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))

in_loader = test_loader(args.batch_size, True, args.in_patches, args.split, args.in_machine)
out_loader = test_loader(args.batch_size, False, args.out_patches, args.split, args.in_machine)
model.outlier_detection(in_loader, out_loader, args.in_patches, args.out_patches)
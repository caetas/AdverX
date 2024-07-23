from data.Dataloader import test_loader
from models.DCGAN import Discriminator
from utils.util import parse_args_DCGAN
import torch

args = parse_args_DCGAN()
model = Discriminator(args.d, 1, 128).to(torch.device('cuda'))

if args.discriminator_checkpoint is not None:
    model.load_state_dict(torch.load(args.discriminator_checkpoint))

in_loader = test_loader(args.batch_size, True, args.in_patches, args.split, args.in_machine)
out_loader = test_loader(args.batch_size, False, args.out_patches, args.split, args.in_machine)
model.outlier_detection(in_loader, out_loader, args.in_patches, args.out_patches)
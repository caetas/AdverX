from data.Dataloader import test_loader
from config import data_processed_dir
from models.AdverX import AdverX
from utils.util import parse_args_AdverX
import torch

args = parse_args_AdverX()
model = AdverX(128,1,args)
model.vae.load_state_dict(torch.load(args.checkpoint))
model.discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))
in_loader = test_loader(args.batch_size, True, args.patches_image, args.split)
out_loader = test_loader(args.batch_size, False, args.patches_image, args.split)
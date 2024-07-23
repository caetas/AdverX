from data.Dataloader import test_loader
from models.Glow import Glow
from utils.util import parse_args_Glow

args = parse_args_Glow()

model = Glow(image_shape        =   (128,128,1), hidden_channels    =   args.hidden_channels, args=args)
model.load_checkpoint(args)
in_loader = test_loader(args.batch_size, True, args.in_patches, args.split, args.in_machine, normalize=False)
out_loader = test_loader(args.batch_size, False, args.out_patches, args.split, args.in_machine, normalize=False)
model.outlier_detection(in_loader, out_loader, args.in_patches, args.out_patches)
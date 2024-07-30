from data.Dataloader import test_loader
from models.Glow import Glow
from utils.util import parse_args_Glow

args = parse_args_Glow()

model = Glow(image_shape        =   (128,128,1), hidden_channels    =   args.hidden_channels, args=args)
model.load_checkpoint(args)

fpr95s = []
aurocs = []
for i in range(args.eval_iters):
    in_loader = test_loader(args.batch_size, True, args.in_patches, args.split, args.in_machine, normalize=False)
    out_loader = test_loader(args.batch_size, False, args.out_patches, args.split, args.in_machine, normalize=False)
    auroc, fpr95 = model.outlier_detection(in_loader, out_loader, args.in_patches, args.out_patches, display=False)
    fpr95s.append(fpr95)
    aurocs.append(auroc)
print(f"Mean AUROC: {sum(aurocs)/args.eval_iters}")
print(f"Mean FPR95: {sum(fpr95s)/args.eval_iters}")

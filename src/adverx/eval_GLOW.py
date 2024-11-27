from data.Dataloader import test_loader
from models.Glow import Glow
from utils.util import parse_args_Glow
import numpy as np

args = parse_args_Glow()

model = Glow(image_shape        =   (128,128,1), hidden_channels    =   args.hidden_channels, args=args)
model.load_checkpoint(args)

n_patches = [1, 16, 32, 64, 128, 256, 512]

for n in n_patches:
    fpr95s = []
    aurocs = []

    for i in range(args.eval_iters):

        assert args.batch_size >= n, "Batch size must be greater than or equal to the number of patches"

        if args.batch_size % n != 0:
            print(f"Warning: Batch size {args.batch_size} is not divisible by the number of patches {n}. Setting batch size to {args.batch_size - args.batch_size % n}")
            args.batch_size = args.batch_size - args.batch_size % n
        
        in_loader = test_loader(args.batch_size, True, n, args.split, args.in_machine, normalize=False)
        out_loader = test_loader(args.batch_size, False, n, args.split, args.in_machine, normalize=False)
        auroc, fpr95 = model.outlier_detection(in_loader, out_loader, n, n, display=False)
        fpr95s.append(fpr95)
        aurocs.append(auroc)
    print(f"Number of patches: {n}")
    print(f"Mean AUROC: {100*sum(aurocs)/args.eval_iters:.2f}")
    print(f"Mean FPR95: {100*sum(fpr95s)/args.eval_iters:.2f}")
    print(f"Two-sigma confidence interval for mean AUROC: {200*np.std(aurocs)/np.sqrt(args.eval_iters):.2f}")
    print(f"Two-sigma confidence interval for mean FPR95: {200*np.std(fpr95s)/np.sqrt(args.eval_iters):.2f}")
    print("\n")

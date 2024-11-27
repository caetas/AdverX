from data.Dataloader import test_loader
from models.VAE import VAE
from utils.util import parse_args_VAE
import torch

args = parse_args_VAE()
model = VAE(128,1,args)

if args.checkpoint is not None:
    print(f"Loading checkpoint from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

n_patches = [1, 16, 32, 64, 128, 256, 512]

for n in n_patches:
    fpr95s = []
    aurocs = []
    for i in range(args.eval_iters):
        in_loader = test_loader(args.batch_size, True, n, args.split, args.in_machine)
        out_loader = test_loader(args.batch_size, False, n, args.split, args.in_machine)
        auroc, fpr95 = model.outlier_detection(in_loader, out_loader, n, n, display=False)
        fpr95s.append(fpr95)
        aurocs.append(auroc)
    print(f"Number of patches: {n}")
    print(f"Mean AUROC: {sum(aurocs)/args.eval_iters}")
    print(f"Mean FPR95: {sum(fpr95s)/args.eval_iters}")
    print("\n")
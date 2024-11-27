from data.Dataloader import test_loader
from models.AdverX import AdverX
from utils.util import parse_args_AdverX
import torch

args = parse_args_AdverX()
model = AdverX(128,1,args)

if args.discriminator_checkpoint is not None:
    print(f"Loading discriminator checkpoint from {args.discriminator_checkpoint}")
    model.discriminator.load_state_dict(torch.load(args.discriminator_checkpoint), strict=False)
    model.discriminator.eval()

batch_sizes = [1, 16, 32, 64, 128, 256, 512]

for batch_size in batch_sizes:

    args.batch_size = batch_size
    args.in_patches = batch_size
    args.out_patches = batch_size

    fpr95s = []
    aurocs = []
    for i in range(args.eval_iters):
        in_loader = test_loader(args.batch_size, True, args.in_patches, args.split, args.in_machine)
        out_loader = test_loader(args.batch_size, False, args.out_patches, args.split, args.in_machine)
        auroc, fpr95 = model.outlier_detection(in_loader, out_loader, args.in_patches, args.out_patches, display = False)
        fpr95s.append(fpr95)
        aurocs.append(auroc)
    # print batch size
    print(f"Batch size: {args.batch_size}")
    print(f"Mean AUROC: {sum(aurocs)/args.eval_iters*100:.2f}")
    print(f"Mean FPR95: {sum(fpr95s)/args.eval_iters*100:.2f}")
    # print confidence interval of 2 sigma for mean AUROC
    print(f"Confidence interval of 2 sigma for mean AUROC: {2*torch.std(torch.tensor(aurocs))/args.eval_iters**0.5*100:.2f}")
    # print confidence interval of 2 sigma for mean FPR95
    print(f"Confidence interval of 2 sigma for mean FPR95: {2*torch.std(torch.tensor(fpr95s))/args.eval_iters**0.5*100:.2f}")
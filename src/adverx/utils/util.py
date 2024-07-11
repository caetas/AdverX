import argparse

def parse_args_AdverX():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
    argparser.add_argument('--hidden_dims', type=int, nargs='+', default=None, help='hidden dimensions')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--gen_weight', type=float, default=0.001, help='generator weight')
    argparser.add_argument('--recon_weight', type=float, default=0.001, help='reconstruction weight')
    argparser.add_argument('--sample_and_save_frequency', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='discriminator checkpoint path')
    argparser.add_argument('--kld_weight', type=float, default=1e-4, help='kl weight')
    argparser.add_argument('--loss_type', type=str, default='mse', help='loss type', choices=['mse', 'ssim'])
    argparser.add_argument('--patches_image', type=int, default=32, help='number of patches per image for training')
    argparser.add_argument('--split', type=float, default=0.7, help='train-test split for the ID dataset')
    argparser.add_argument('--in_machine', type=str, default='siemens', help='in-distribution machine', choices=['siemens', 'ge', 'philips', 'gmm', 'konica'])
    argparser.add_argument('--in_patches', type=int, default=8, help='number of patches per image for in-distribution evaluation')
    argparser.add_argument('--out_patches', type=int, default=2, help='number of patches per image for out-of-distribution evaluation')
    return argparser.parse_args()

# EOF

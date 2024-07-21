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


def parse_args_DCGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lrg', type=float, default=0.0002, help='learning rate generator')
    argparser.add_argument('--lrd', type=float, default=0.0002, help='learning rate discriminator')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    argparser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    argparser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    argparser.add_argument('--img_size', type=int, default=32, help='image size')
    argparser.add_argument('--channels', type=int, default=1, help='channels')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample interval')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='discriminator checkpoint path')
    argparser.add_argument('--n_samples', type=int, default=9, help='number of samples')
    argparser.add_argument('--d', type=int, default=64, help='d')
    argparser.add_argument('--patches_image', type=int, default=32, help='number of patches per image for training')
    argparser.add_argument('--split', type=float, default=0.7, help='train-test split for the ID dataset')
    argparser.add_argument('--in_machine', type=str, default='siemens', help='in-distribution machine', choices=['siemens', 'ge', 'philips', 'gmm', 'konica'])
    argparser.add_argument('--in_patches', type=int, default=8, help='number of patches per image for in-distribution evaluation')
    argparser.add_argument('--out_patches', type=int, default=2, help='number of patches per image for out-of-distribution evaluation')
    return argparser.parse_args()


def parse_args_PresGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    argparser.add_argument('--ngf', type=int, default=64)
    argparser.add_argument('--ndf', type=int, default=64)

    ###### Optimization arguments
    argparser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    argparser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    argparser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    argparser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')

    ###### Checkpointing and Logging arguments
    argparser.add_argument('--checkpoint', type=str, default=None, help='a given checkpoint file for generator')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='a given checkpoint file for discriminator')
    argparser.add_argument('--sigma_checkpoint', type=str, default=None, help='a given file for logsigma for the generator')
    argparser.add_argument('--num_gen_images', type=int, default=16, help='number of images to generate for inspection')

    ###### PresGAN-specific arguments
    argparser.add_argument('--sigma_lr', type=float, default=0.0002, help='generator variance')
    argparser.add_argument('--lambda_', type=float, default=0.01, help='entropy coefficient')
    argparser.add_argument('--sigma_min', type=float, default=0.01, help='min value for sigma')
    argparser.add_argument('--sigma_max', type=float, default=0.3, help='max value for sigma')
    argparser.add_argument('--logsigma_init', type=float, default=-1.0, help='initial value for log_sigma_sian')
    argparser.add_argument('--num_samples_posterior', type=int, default=2, help='number of samples from posterior')
    argparser.add_argument('--burn_in', type=int, default=2, help='hmc burn in')
    argparser.add_argument('--leapfrog_steps', type=int, default=5, help='number of leap frog steps for hmc')
    argparser.add_argument('--flag_adapt', type=int, default=1, help='0 or 1')
    argparser.add_argument('--delta', type=float, default=1.0, help='delta for hmc')
    argparser.add_argument('--hmc_learning_rate', type=float, default=0.02, help='lr for hmc')
    argparser.add_argument('--hmc_opt_accept', type=float, default=0.67, help='hmc optimal acceptance rate')
    argparser.add_argument('--stepsize_num', type=float, default=1.0, help='initial value for hmc stepsize')
    argparser.add_argument('--restrict_sigma', type=int, default=0, help='whether to restrict sigma or not')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    return argparser.parse_args()

# EOF

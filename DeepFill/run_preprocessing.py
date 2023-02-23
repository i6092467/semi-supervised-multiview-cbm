import os
import glob

import subprocess

from .cv_utils import prepare_inpainting
from .tester import WGAN_tester

def deep_fill(
            data_dir, # Raw dataset directory
            test_dir, # Test data directory for DeepFill
            mask_dir, # Mask directory for DeepFill
            results_dir, # Output directory for DeepFill
            template_dir # Directory with pattern templates to be removed
            ):

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Empty relevant folders
    files = glob.glob(mask_dir + '/*')
    for f in files:
        os.remove(f)
    files = glob.glob(test_dir + '/*')
    for f in files:
        os.remove(f)
    files = glob.glob(results_dir + '/*')
    for f in files:
        os.remove(f)

    # Prepare for image inpainting: create masks, copy images...
    print()
    print()
    print('Preparing inpainting...')
    prepare_inpainting(image_dir=data_dir, test_dir=test_dir, mask_dir=mask_dir, template_dir=template_dir)
    
    class Object(object):
        pass
    opt = Object()
    opt.results_path = results_dir
    opt.gan_type = 'WGAN'
    opt.gpu_ids = '1'
    opt.cudnn_benchmark = True
    opt.epoch = 40
    opt.batch_size = 1
    opt.num_workers = 0
    opt.in_channels = 4
    opt.out_channels = 3
    opt.latent_channels = 48
    opt.pad_type = 'zero'
    opt.activation = 'elu'
    opt.norm = 'none'
    opt.init_type = 'xavier'
    opt.init_gain = 0.02
    opt.baseroot = test_dir
    opt.baseroot_mask = mask_dir

    print('Applying DeepFill...')
    WGAN_tester(opt)



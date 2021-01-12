from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Paint')
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--mode', help='task to be done', default='paint2image')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:

            N = len(reals) - 1
            n = opt.paint_start_scale
   
            if opt.quantization_flag:
                opt.mode = 'paint_train'
                dir2trained_model = functions.generate_dir2save(opt)
                # N = len(reals) - 1
                # n = opt.paint_start_scale
                real_s = imresize(real, pow(opt.scale_factor, (N - n)), opt)
                real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                real_quant, centers = functions.quant(real_s, opt.device)
                plt.imsave('%s/real_quant.png' % dir2save, functions.convert_image_np(real_quant), vmin=0, vmax=1)
                if (os.path.exists(dir2trained_model)):
                    # print('Trained model does not exist, training SinGAN for SR')
                    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
                    opt.mode = 'paint2image'
                else:
                    train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, opt.paint_start_scale)
                    opt.mode = 'paint2image'

# -*- coding: utf-8 -*-


import torch as th
import torch.nn as nn
th.manual_seed(0)
import json
import torch.nn.functional as F
import cv2
device_ids = [0]
from tqdm import tqdm
import scipy.misc
import timeit
from utils.data_utils import *
from utils.model_utils import *
import gc
from models.encoder.encoder import FPNEncoder
import argparse
import numpy as np
import os
import torch.optim as optim
from torchvision import transforms
import lpips as lpips
from utils.mask_manipulate_utils import *
import imageio
from models.EditGAN import *
from utils.data_utils import face_palette as palette
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

import sys
import glob



class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def embed_one_example(args, path, stylegan_encoder, g_all, upsamplers,
                      inter, percept, steps, sv_dir,
                      skip_exist=False):

    if os.path.exists(sv_dir):
        if skip_exist:
            return 0,0,[], []
        else:
            pass
    else:
        os.system('mkdir -p %s' % (sv_dir))
    print('SV folder at: %s' % (sv_dir))
    image_path = path
    label_im_tensor, im_id = load_one_image_for_embedding(image_path, args['im_size'])

    print("****** Run optimization for ", path, " ******")


    label_im_tensor = label_im_tensor.to(device)
    label_im_tensor = label_im_tensor * 2.0 - 1.0
    label_im_tensor = label_im_tensor.unsqueeze(0)
    latent_in = stylegan_encoder(label_im_tensor)
    im_out_wo_encoder, _ = latent_to_image(g_all, upsamplers, latent_in,
                                           process_out=True, use_style_latents=True,
                                           return_only_im=True)

    args['use_noise'] = False
    args['noise_loss_weight'] = 100

    out = run_embedding_optimization(args, g_all,
                                     upsamplers, inter, percept,
                                     label_im_tensor, latent_in, steps=steps,
                                     stylegan_encoder=stylegan_encoder,
                                     use_noise=args['use_noise'],
                                     noise_loss_weight=args['noise_loss_weight']
                                     )
    if args['use_noise']:
        optimized_latent, optimized_noise, loss_cache = out
        optimized_noise = [torch.from_numpy(noise).cuda() for noise in optimized_noise]
    else:
        optimized_latent, optimized_noise, loss_cache = out
        optimized_noise = None
    print("Curr loss, ", loss_cache[0], loss_cache[-1] )

    optimized_latent_np = optimized_latent.detach().cpu().numpy()[0]
    if args['use_noise']:
        loss_cache_np = [noise.detach().cpu().numpy() for noise in optimized_noise]
    else:
        loss_cache_np = []
    # vis
    img_out, _ = latent_to_image(g_all, upsamplers, optimized_latent,
                                 process_out=True, use_style_latents=True,
                                 return_only_im=True, noise=optimized_noise)

    raw_im_show = (np.transpose(label_im_tensor.cpu().numpy(), (0, 2, 3, 1))) * 255.
    raw_im_show = raw_im_show.astype(np.uint8)
    
    vis_list = [im_out_wo_encoder[0], img_out[0]
                ]

    curr_vis = np.concatenate(
        vis_list, 0)

    print(f'Shape curr_vis: {curr_vis.shape}')
    print(f'Shape raw_im_show: {raw_im_show[0].shape}')


    return optimized_latent


def run_seg(args, g_all, bi_upsamplers, classifier_list,  optimized_latent, index):

        img_out, affine_layers = latent_to_image(g_all, bi_upsamplers, optimized_latent, process_out=True,
                                                 return_upsampled_layers=False,
                                                 use_style_latents=True, dim=512, return_only_im=False) #dim=256
        image_features = []
        for i in range(len(affine_layers)):
            image_features.append(bi_upsamplers[i](
                affine_layers[i]))
        image_features = torch.cat(image_features, 1)
        print(f'Shape image_features: {image_features.shape}')
        #np.save(os.path.join(args['exp_dir'], "image_features.npy"),image_features.cpu())
        #print('Features saved!')
        #image_features = image_features[:, :, 64:448]
        image_features = image_features[:, 0:6016]
        #print(f'Shape image_features 2: {image_features.shape}')
        #image_features = image_features[0]
        if args['use_intermediate_layers']:
            intermediateLayer = torch.from_numpy(np.load(os.path.join( args['intermediate_layer'], f'intermediateLayers_{str(index).zfill(4)}_0150.npy'))).to(device)
            intermediateLayer = intermediateLayer.unsqueeze(0)
            #intermediate_layer_max, _ = torch.max(intermediateLayer.abs(), dim=3)
            #intermediate_layer_max = intermediate_layer_max.unsqueeze(3)
            print(f"intermediateLayer: {os.path.join( args['intermediate_layer'], f'intermediateLayers_{str(index).zfill(4)}_0150.npy')}")
            print(f'intermediateLayer shape: {intermediateLayer.shape}')
            image_features = image_features.permute(0, 2, 3, 1)
            
            image_features = torch.cat((image_features, intermediateLayer), dim=3)
            print(f'image_features concat shape: {image_features.shape}')

        else:    
            image_features = image_features.permute(0, 2, 3, 1)

        print(f'Shape image_features 2: {image_features.shape}')
        image_features = image_features.reshape(-1, args['dim_prep_data'][2]) #torch.Size([196608, 6016])
        print(f'Shape image_features 3: {image_features.shape}')
        #np.save(os.path.join(args['exp_dir'], "image_features.npy"),image_features.cpu())
        #image_features = image_features.reshape(args['dim'], -1).transpose(1, 0)
        seg_mode_ensemble = []
        for MODEL_NUMBER in range(args['num_classifier']):
            classifier = classifier_list[MODEL_NUMBER]
            img_seg = classifier(image_features)
            seg_mode_ensemble.append(img_seg.unsqueeze(0))
        print(f'Shape img_seg: {img_seg.shape}')
        img_seg_final = torch.argmax(torch.mean(torch.cat(seg_mode_ensemble, 0), 0),1).reshape(256, 256).detach().cpu().numpy()
        #img_seg_final = torch.argmax(torch.mean(torch.cat(seg_mode_ensemble, 0), 0),1).reshape(384, 512).detach().cpu().numpy()
        del (affine_layers)
        img_out =  cv2.resize(np.squeeze(img_out[0]), dsize=(args['dim_prep_data'][1], args['dim_prep_data'][1]), interpolation=cv2.INTER_NEAREST)
        return img_out, img_seg_final

def main(args):

    latent_sv_folder = args['optimized_latent_path']['train']
    steps = args['embedding_steps']
    sv_folder = latent_sv_folder


    assert latent_sv_folder != ""
    all_images = []
    all_id = []

    curr_images_all = glob.glob(args['testing_data_path'] +  "*/*")
    curr_images_all = [data for data in curr_images_all if ('jpg' in data or 'webp' in data or 'png' in data  or 'jpeg' in data or 'JPG' in data) and not os.path.isdir(data)  and not 'npy' in data ]
    
    splits = KFold(n_splits = 4, shuffle = False)

    for i, image in enumerate(curr_images_all):
        all_id.append(image.split("/")[-1].split(".")[0])
        all_images.append(image)

    valid_idx = np.load(os.path.join(args['exp_dir'],"ids_val.npy"), allow_pickle=True).item()
    print(valid_idx)
    print("All files, " , len(all_images))

    for fold in valid_idx.keys():
        print(f"FOLD {fold}, Val ids: {valid_idx[fold]}")

        g_all, nn_upsamplers, bi_upsamplers, classifier_list, avg_latent = prepare_model(args,classfier_checkpoint_path=args['classfier_checkpoint'],classifier_iter=args['classifier_iter'], num_class=args['num_class'], num_classifier=args['num_classifier'], num_fold=fold) #classifier_iter=10000
        inter = Interpolate(args['im_size'][1], 'bilinear')

        percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), normalize=args['normalize']
        ).to(device)
        n_latent =  g_all.module.n_latent
        stylegan_encoder = FPNEncoder(3, n_latent=n_latent, only_last_layer=args['use_w'])
        stylegan_encoder = stylegan_encoder.to(device)
        stylegan_encoder.load_state_dict(torch.load(args['encoder_checkpoint'], map_location=device)['model_state_dict'], strict=True)

        
        for i in valid_idx[fold]:
            optimized_latent = embed_one_example(args, all_images[i],
                                                stylegan_encoder, g_all,
                                                bi_upsamplers, inter, percept, steps,
                                                sv_folder, skip_exist=False)

            print(f'Optimized_latent shape: {optimized_latent.shape}')
            img_out, img_seg_final = run_seg(args, g_all, bi_upsamplers, classifier_list, optimized_latent, i)
            
            print(f'Shape img_out: {img_out.shape}')
            imageio.imsave(os.path.join(args['exp_dir'], f"evaluate_img_out_{i}_fold_{fold}.jpg"), img_out.astype(np.uint8))
            imageio.imsave(os.path.join(args['exp_dir'], f"evaluate_img_seg_final_{i}_fold_{fold}.png"), img_seg_final.astype(np.uint8))
        #all_feature_maps, all_mask, num_data = prepare_data(args, palette)
        
        print(' -- Data prepared -- ')

        #print(f'Shape img_seg_final: {img_seg_final.shape}')
        #sys.exit()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='')
    
    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))

    main(args=opts)
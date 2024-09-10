"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys

import torch as th
import torch.distributed as dist

import cv2
import matplotlib.pyplot as plt

import scipy
import numpy as np


from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)



def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(model_and_diffusion_defaults())
    logger.log("model and diffusion created")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    logger.log("sending weights to GPU...")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    datasetImages = os.listdir(args.path_dataset)
    datasetImages.sort()

    logger.log(f"creating {args.batch_size} batch samples")
    all_images = []
    all_labels = []
    i = 0

    while i < len(datasetImages):    
        #logger.log("load models...")
        model_kwargs = {}
        if args.class_cond:
            if args.dataset == '':
                classes = th.from_numpy(np.array([args.imagenet_class], dtype=np.int64)).to(dist_util.dev())
                logger.log(f'Image class choosen: {classes}')
            else:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.modified != True:
            logger.log('Code not modified')
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            logger.log('Sample')
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        else:
            #logger.log('Code modified')
            args.image_to_segment = f'{args.path_dataset}/{datasetImages[i]}'
            #diffusion.quitaruido = True
            #diffusion.always_save = True
            index = args.timestep_to_segment
            num_timesteps = int(args.timestep_respacing)
            i_noise=num_timesteps-index
            q_noise=num_timesteps-index
            log_noise=-1.5
            i_unet=num_timesteps-index
            
            img = cv2.imread(args.image_to_segment)
            imgResized = cv2.resize(img, (args.image_size,args.image_size))

            imgResizedT = imgResized.transpose((2, 0, 1))[::-1]
            
            sampleDict  = diffusion.p_sample_any_image(
                model,
                (1, 3, args.image_size, args.image_size),
                imgResizedT,
                i_unet,
                index,
                i_noise=None,
                #log_noise=log_noise,
                q_noise=q_noise,
                noise=None,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            
            sample = sampleDict['sample']
            intermediateLayers = sampleDict['intermediateLayers']
            samplePred_xstart = sampleDict['pred_xstart']

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            
            imgSave = sample.squeeze(0).cpu().numpy()

            #Intermediate layers
            imgLayers = sample.squeeze(0).cpu().numpy().transpose(2,0,1)

            cv2.imwrite(f'{args.save_results}/image_out_{i:04d}.png', cv2.cvtColor(imgSave, cv2.COLOR_RGB2BGR))
            np.savez(f'{args.save_results}/intermediateLayers_{i:04d}_{args.timestep_to_segment:04d}.npz', **intermediateLayers)
            
            all_images.extend([sample.cpu().numpy()])
            i += 1

        
    logger.log(f"created {len(all_images) * args.batch_size} samples")
    
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

def concatenateIntermediateLayers(args, intermediateLayers: dict):
    del intermediateLayers['t']
    del intermediateLayers['index']

    upsampled_layers = []
    namesLayers = list(intermediateLayers.keys())
    
    for layer in namesLayers:
        if layer in (args.intermediate_layers_saved).split():
            multFactor = 256 / intermediateLayers[layer].squeeze().shape[2]
            #layer_upsampled = scipy.ndimage.zoom(intermediateLayers[layer].squeeze(), (1,multFactor,multFactor), order=1)
            layer_upsampled = scipy.ndimage.zoom(intermediateLayers[layer].squeeze()[0:4], (1,multFactor,multFactor), order=1)
            #upsampled_layers.append(layer_upsampled.reshape(1,256,256))
            upsampled_layers.append(layer_upsampled)

    layers_concatenated = np.concatenate(upsampled_layers, axis=0)
    return layers_concatenated

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=2,
        use_ddim=False,
        model_path="",
        imagenet_class=817,
        class_cond = False,
        modified = True,
        timestep_to_segment = 250,
        timestep_respacing = 250,
        image_to_segment = "",
        dataset = "",
        intermediate_layers_saved = "h_m h_o02 h_o07 h_o14 h_o18 h_o23",
        path_dataset = "",
        save_results = "",
        is_train = False
    )
    
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

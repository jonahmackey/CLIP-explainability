import os
import random
import math
import kornia as K
from typing import Optional
import numpy as np
import matplotlib as plt
import pandas
import argparse
from datetime import datetime

import torch
from torchvision import io, utils
from torch.autograd.functional import jacobian
from torch.fft import *

from clip import load, tokenize


def rand_cutout(image, size, center_bias=False, center_focus=2):
    width = image.shape[-1]
    min_offset = 0
    max_offset = width - size
    if center_bias:
        # sample around image center
        center = max_offset / 2
        std = center / center_focus
        offset_x = int(random.gauss(mu=center, sigma=std))
        offset_y = int(random.gauss(mu=center, sigma=std))
        # resample uniformly if over boundaries
        offset_x = random.randint(min_offset, max_offset) if (offset_x > max_offset or offset_x < min_offset) else offset_x
        offset_y = random.randint(min_offset, max_offset) if (offset_y > max_offset or offset_y < min_offset) else offset_y
    else:
        offset_x = random.randint(min_offset, max_offset)
        offset_y = random.randint(min_offset, max_offset)
    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
    return cutout


def get_cutouts(img, num_cutouts, legal_cutouts):
    """
    Helper function to get num_cutouts random cutouts from img. Each random cutouts is resized to (224, 224) after being
    sampled.

    Args:
        img (torch.Tensor): image to sample cutouts from, tensor with shape (1, 3, H, W)
        num_cutouts (int): number of cutouts to be sampled
        legal_cutouts (torch.Tensor): list of legal cutout sizes

    Returns:
        torch.Tensor: tensor with shape (num_cutouts, 3, H, W) containing image cutouts
    """

    if num_cutouts == 0:
        resized_img = K.geometry.resize(img, (224, 224), antialias=True) # shape (1, 3, 224, 224)
        return resized_img

    cutouts = []

    for i in range(num_cutouts):
        # get legal cutout size
        size = int(512 * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        size = legal_cutouts[torch.argmin(torch.abs(legal_cutouts - size))].cpu().item()

        # get random cutout of given size
        random_cutout = rand_cutout(img, size, center_bias=False).cuda()  # shape (1, 3, size, size)

        # up/down sample to 224x224
        random_cutout = K.geometry.resize(random_cutout, (224, 224), antialias=True)  # shape (1, 3, 224, 224)

        cutouts.append(random_cutout)

    cutouts = torch.cat(cutouts)  # shape (num_cutouts, 3, 224, 224)

    return cutouts

    
def embed_image(img, img_fft_abs=1, eps=1e-6, num_cutouts=32, freq_reg=None):
    """
    Embed img into CLIP.

    Args:
        img (torch.Tensor): input image with shape (1, 3, H, W) and values in range [0, 1]
        img_fft_abs (int): term for regularization. Defaults to 1.
        eps (float): _description_. Defaults to 1e-6.
        num_cutouts (int): number of cutouts to be sampled from img. Defaults to 32.
        freq_reg (str, optional): str in {None, 'norm', 'log'} indicating image regularization. Defaults to None.

    Returns:
        torch.Tensor: embedding of img in CLIP, shape (512)
    """
    
    # apply regularization
    if freq_reg == 'norm':
        img = irfft2((img_fft_abs - eps) * rfft2(img))  # shape (1, 3, 512, 512)
    elif freq_reg == 'log':
        # img = irfft2(torch.exp())
        img = img
    else:  # freq_reg = None
        img = img

    # img has shape 1x3xHxW
    im_size = img.shape[2]

    # define legal cutout sizes
    legal_cutouts = torch.arange(start=1, end=16, step=1, dtype=torch.float32).cuda()
    legal_cutouts = torch.round((im_size * 7) / (7 + legal_cutouts)).int()

    # sample cutouts from image and normalize
    image_into = get_cutouts(img=img,
                             num_cutouts=num_cutouts,
                             legal_cutouts=legal_cutouts)  # shape (num_cutouts, 3, 224, 224)
    image_into = normalize_image(image_into)   
    
    # embed image into clip
    image_embed = perceptor.encode_image(image_into)  # shape (num_cutouts, 512)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed = torch.mean(image_embed, dim=0)  # shape (512)

    return image_embed


def get_embedding_function(img_fft_abs=1, eps=1e-6, num_cutouts=32, freq_reg=None):
    """
    Helper function that returns an embedding_function with some frozen inputs. This
    allows us to take the jacobian of embedding_function with respect to only the img
    parameter.
    """
    def embedding_func(img):
        return embed_image(img, img_fft_abs, eps, num_cutouts, freq_reg)

    return embedding_func


def jacob_svd(jacob, save_path, sv_indices=[], save_results=True):
    """
    Compute singular value decomposition of jacobian,
        J = U * diag(S) * V^T,
    and save visualizations of singular values and pre-image of left singular vectors.

    Args:
        title (str): save file path
        jacob (torch.tensor): a jacibian matrix, tensor with shape (512, 3, H, W)
        sv_indices (list, optional): list of indices of the singular vectors we want to visualize. Defaults to [].

    Returns:
        tuple[torch.Tensor]: U, S, and V^T
    """
    im_size = jacob.shape[2]

    jacob = jacob.cpu().detach()
    jacob = jacob.view(512, 3 * im_size * im_size).numpy()  # shape (512, 3*im_size*im_size)

    (U, S, Vt) = np.linalg.svd(jacob, full_matrices=False)

    # U.shape = (512, 512)
    # S.shape = (512)
    # Vt.shape = (512, 3*im_size*im_size)

    if save_results:
        # plot singular values
        x = np.arange(start=1, stop=S.shape[0] + 1, step=1)

        fig = plt.figure(figsize=(20, 10))

        # singular values
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(x, S, marker=".")

        ax1.set_ylabel("Value")
        ax1.set_xlabel("Rank")
        plt.title("Singular Values of Jacobian")

        # singular values (log)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(x, S, marker=".")
        ax2.set_yscale('log')

        ax2.set_ylabel("Value (log)")
        ax2.set_xlabel("Rank")
        plt.title("Singular Values of Jacobian (log)")
        
        plt.savefig(save_path + "/SV_jacob/sing_vals.png")
        
        # visualize and save the preimage of the left singular vectors in pixel space
        U = torch.from_numpy(U)
        
        for sv_index in sv_indices:
            singular_value = U[:, sv_index] # shape (512)
            sv_preimage = jacob.t() @ singular_value # shape (3*im_size*im_size)
            sv_preimage.view(3, im_size, im_size) # shape (3, im_size, im_size)
            
            vmin, vmax = torch.quantile(input=sv_preimage.abs(), q=torch.tensor([0., 1.]))
            
            # Plot the RGB channels of sv preimage separately (yellow = 1, purple = 0)
            fig = plt.figure(figsize=(30, 10))

            fig.add_subplot(1, 3, 1)
            plt.axis("off")
            plt.title("red")
            plt.imshow(sv_preimage[0], vmin=-vmax, vmax=vmax, cmap='PiYG')
            plt.colorbar(shrink=0.5)

            fig.add_subplot(1, 3, 2)
            plt.axis("off")
            plt.title("green")
            plt.imshow(sv_preimage[1], vmin=-vmax, vmax=vmax, cmap='PiYG')
            plt.colorbar(shrink=0.5)

            fig.add_subplot(1, 3, 3)
            plt.axis("off")
            plt.title("blue")
            plt.imshow(sv_preimage[2], vmin=-vmax, vmax=vmax, cmap='PiYG')
            plt.colorbar(shrink=0.5)
            
            plt.savefig(save_path + f"/SV_jacob/sv_preimage{sv_index + 1}.png")

    return U, S, Vt



def push_img_to_embedding(img, 
                          tgt_embed, 
                          tgt_num, 
                          data_list, 
                          save_every, 
                          threshold, 
                          lr, 
                          max_iters, 
                          num_cutouts):

    img.requires_grad = True
    optimizer = torch.optim.SGD(params=[img], lr=lr)

    step = 0
    loss = 10
    took_max_steps = False

    embed_image_func = get_embedding_function(freq_reg=False, num_cutouts=num_cutouts)

    while loss > threshold:

        # embed image
        embed = embed_image_func(img).float()

        # compute loss
        loss = torch.norm(input=embed - tgt_embed, p='fro')

        # update image pixels
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # increment iter steps
        step += 1

        # collect data
        if step % save_every == 1:
            data_list.append([tgt_num, step, loss.item(), img.max().item(), img.min().item(), img.grad.norm().item()])
        
        # break loop after reaching max_iters iters
        if step == max_iters:
            took_max_steps = True
            data_list.append([tgt_num, step, loss.item(), img.max().item(), img.min().item(), img.grad.norm().item()])
            print("break optimization")
            break
    
    print(f"Optimized img in {step} steps, with final loss={loss}")
    return img, data_list, took_max_steps


def clip_dream(img_fp,
               exp_name,
               save_date,
               theta,
               sv_index,
               rand_dir,
               threshold,
               eps,
               num_cutouts,
               freq_reg,
               lr,
               max_iters):
    """
    Creates a CLIP dream.
    
    Given an image I, we compute the jacobian of the embedding of I in CLIP, E = f(I), with respect to 
    the frequency regularized pixels I:
        J = df(I)/dI
    We then compute the singular value decomposition of J:
        J = U * diag(S) * V^T
    The first 511 columns of U live in the tangent space of E, each corresponding to a singular value that 
    indicates this
    """
    if save_date:
        save_path = "./Results/" + datetime.now().strftime("%y%m%d-%H%M%S-") + exp_name 
    else:
        save_path = "./Results/" + exp_name 
    
    try:
        os.makedirs(save_path)
    except:
        pass
    
    img = io.read_image(img_fp).to(device)  # shape (3, 512, 512)
    img = img / 255  # put pixels in range [0, 1]
    img = torch.unsqueeze(img, dim=0).float()  # shape (1, 3, 512, 512)

    utils.save_image(img.float().squeeze(), save_path + f"/dream_iter{0:>04d}.png")

    if freq_reg == 'norm':
        img_fft = rfft2(img)  # shape (1, 3, 512, 257)
        img_fft_abs = img_fft.abs()  # shape (1, 3, 512, 257)
        input_img = irfft2(img_fft / (img_fft_abs + eps))  # shape (1, 3, 512, 512)
    else:
        input_img = img.clone()
        img_fft_abs = 1

    input_img.requires_grad = True

    # get embedding func
    embed_image_func = get_embedding_function(img_fft_abs=img_fft_abs, eps=eps, num_cutouts=num_cutouts,
                                              freq_reg=freq_reg)

    # jacobian of image embedding wrt reqularized img pixels
    jacob = jacobian(func=embed_image_func, inputs=input_img).squeeze()  # shape (512, 3, 512, 512)

    # get non regularized image embedding
    standard_embed_func = get_embedding_function(freq_reg=False, num_cutouts=num_cutouts)
    image_embedding = standard_embed_func(img)  # shape (512)

    # get tangent vector to image_embedding
    if rand_dir: # get random tangent vector
        v_rand = torch.rand(512).half().cuda()
        v_tang = v_rand - (image_embedding @ v_rand) * image_embedding
        v_tang = v_tang / (v_tang.norm(dim=-1, keepdim=True))

    else: # get singular vector
        # singular value decomposition
        U, S, Vt = jacob_svd(save_path=save_path,
                             jacob=jacob,
                             sv_indices=[sv_index])

        v_tang = U[:, sv_index].cuda()  # shape (512)

    # create dream in forward direction (direction of v_tang)
    data_list = [] # target #, step #, step loss, pixel max, pixel min, gradient norm
    clip_dream_frames1 = []
    tgt_not_reached = 0
    tgt_num = 1
    img_dir1 = img.clone()
    
    print("Starting CLIP dream...")
    while tgt_not_reached < 5:
        # get target
        target = math.cos(tgt_num * math.radians(theta)) * image_embedding + math.sin(tgt_num * math.radians(theta)) * v_tang

        # optimize img 
        img_dir1, data_list, took_max_steps = push_img_to_embedding(img=img_dir1, 
                                                                    tgt_embed=target,
                                                                    tgt_num=tgt_num,
                                                                    data_list=data_list,
                                                                    threshold=threshold,
                                                                    lr=lr,
                                                                    max_iters=max_iters,
                                                                    num_cutouts=num_cutouts)

        print(f"Reached point {tgt_num}")
        
        # if target was not reached before reaching max iters, increment took_max_steps
        if took_max_steps:
            tgt_not_reached += 1
            
        # ensures that we count the number of intermediate targets not reached
        elif (not took_max_steps) and tgt_not_reached > 0:
            tgt_not_reached = 0
            
        tgt_num += 1

        # save clipped image
        img_clipped = torch.clamp(input=img_dir1, min=0, max=1)
        clip_dream_frames1.append(img_clipped.permute(0, 2, 3, 1)) # append img with shape (1, H, W, C)
        utils.save_image(img_clipped.float().squeeze(), save_path + f"/CLIP_dream/dream_iter{tgt_num:>04d}.png")
        
    # create dream in backward direction (direction of -v_tang)
    clip_dream_frames2 = []
    tgt_not_reached = 0
    tgt_num = -1
    img_dir2 = img.clone()
    
    while tgt_not_reached < 5:
        # get target
        target = math.cos(tgt_num * math.radians(theta)) * image_embedding + math.sin(tgt_num * math.radians(theta)) * v_tang

        # optimize img 
        img_dir2, data_list, took_max_steps = push_img_to_embedding(img=img_dir2, 
                                                                    tgt_embed=target,
                                                                    tgt_num=tgt_num,
                                                                    data_list=data_list,
                                                                    threshold=threshold,
                                                                    lr=lr,
                                                                    max_iters=max_iters,
                                                                    num_cutouts=num_cutouts)

        print(f"Reached point {tgt_num}")
        
        # if target was not reached before reaching max iters, increment took_max_steps
        if took_max_steps:
            tgt_not_reached += 1
            
        # ensures that we count the number of intermediate targets not reached
        elif (not took_max_steps) and tgt_not_reached > 0:
            tgt_not_reached = 0
            
        tgt_num -= 1

        # save clipped image
        img_clipped = torch.clamp(input=img_dir2, min=0, max=1)
        clip_dream_frames2.append(img_clipped.permute(0, 2, 3, 1)) # append img with shape (1, H, W, C)
        utils.save_image(img_clipped.float().squeeze(), save_path + f"CLIP_dream/dream_iter_neg{-tgt_num:>04d}.png")
        
    df = pandas.DataFrame(data_list, columns=["Target #", "Step #", "Step loss", "Pixel max", "Pixel min", "Gradient norm"])
    df.to_csv(save_path + f"/experiment_data.csv")
    
    clip_dream_frames2.reverse()
    
    clip_dream_video = clip_dream_frames2 + [img.permute(0, 2, 3, 1)] + clip_dream_frames1
    clip_dream_video = torch.cat(clip_dream_video, dim=0) # shape (N, H, W, C)
    io.write_video(filename=save_path + "/clip_dream_vid.mp4")

    # # create video
    # if make_vid:
    #     make_video(frames_path=title)
    
    print(f"Optimization is done! View your CLIP dream at: {save_path}/clip_dream.mp4")
            

def make_video(frames_path, fps=5):
    os.system(f"ffmpeg -framerate {fps} -i ./Images/{frames_path}/dream_iter%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./Images/{frames_path}/clip_dream.mp4")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_encoder', type=str, default='ViT-B/32', 
                        help='the CLIP image encoder model. Options are "ViT-B/32" or "RN101".')
    
    parser.add_argument('--exp_name', type=str, default='exp1', 
                        help='the name of the experiment, used to name the returned files.')
    parser.add_argument('--img_fp', type=str, default='Images/bird.jpg', 
                        help='file path to image used for CLIP dream.')
    
    parser.add_argument('--sv_index', type=int, default=0, 
                        help='index of singular vector used for CLIP dream.')
    parser.add_argument('--rand_dir', type=bool, default=False, 
                        help='use random direction for CLIP dream instead of singular vector. If True, sv_index will be ignored.')
    parser.add_argument('--theta', type=int, default=1, 
                        help='value of theta, the separation angle between optimization points.')
    parser.add_argument('--threshold', type=float, default=0.1, 
                        help='loss threshold used to determine when to save image start optimization towards next point.')
    parser.add_argument('--max_iters', type=int, default=1000, 
                        help='maximum number of optimization iterations toward point before breaking optimization.')
    parser.add_argument('--eps', type=float, default=1e-6, 
                        help='division term for regularazation methods to avoid division by zero.')
    parser.add_argument('--num_cutouts', type=int, default=32, 
                        help='number of image cutouts used to embed the image into CLIP. When num_cutouts is zero, a simple resize of the image is used.')
    parser.add_argument('--lr', type=float, default=250, 
                        help='learning rate used for optimization.')
    parser.add_argument('--freq_reg', type=Optional[str], default=None, 
                        help='frequency regularization method used for image pixels in jacobian. Options are None, "norm", and "log".')
    
    args = parser.parse_args()
    
    # load CLIP model
    perceptor, normalize_image = load(args.img_encoder, jit=False)

    clip_dream(img_fp=args.img_fp,
               exp_name=args.exp_name,
               save_date=True,
               theta=args.theta,
               sv_index=args.sv_index,
               rand_dir=args.rand_dir,
               threshold=args.threshold, 
               eps=args.eps,
               num_cutouts=args.num_cutouts,
               freq_reg=args.freq_reg,
               lr=args.lr,
               max_iters=args.max_iters)
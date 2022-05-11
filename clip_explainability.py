import os
import random
import math
import kornia as K
from typing import Optional
import numpy as np
import matplotlib as plt
import pandas

import torch
from torchvision import io, utils
from torch.autograd.functional import jacobian
from torch.fft import *

from clip import load, tokenize

# load CLIP model
perceptor, normalize_image = load("ViT-B/32", jit=False)

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

    if num_cutouts == -1:
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


def jacob_svd(title, jacob, sv_indices=[]):
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
    
    plt.savefig(f"Images/{title}/SV_jacob.png")
    
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
        
        plt.savefig(f"Images/{title}/sv_preimage{sv_index+1}.png")

    return U, S, Vt


def clip_dream(img_fp: str,
               title: str,
               theta: int = 1,
               sv_index: int = 0,
               threshold: float = 0.1,
               eps: float = 1e-6,
               num_cutouts: int = 32,
               freq_reg: Optional[str] = 'norm',
               lr: float = 10,
               max_iters: int = 10,
               root: bool = True,
               rand_direction: bool = False,
               make_vid: bool = False
               ):
    """
    Creates a CLIP dream.
    
    Given an image I, we compute the jacobian of the embedding of I in CLIP, E = f(I), with respect to 
    the frequency regularized pixels I:
        J = df(I)/dI
    We then compute the singular value decomposition of J:
        J = U * diag(S) * V^T
    The first 511 columns of U live in the tangent space of E, each corresponding to a singular value that 
    indicates this

    Args:
        img_fp (str): _description_
        title (str): _description_
        theta (int, optional): _description_. Defaults to 1.
        sv_index (int, optional): _description_. Defaults to 0.
        threshold (float, optional): _description_. Defaults to 0.1.
        eps (float, optional): _description_. Defaults to 1e-6.
        num_cutouts (int, optional): _description_. Defaults to 32.
        freq_reg (Optional[str], optional): _description_. Defaults to 'norm'.
        lr (float, optional): _description_. Defaults to 10.
        max_iters (int, optional): _description_. Defaults to 10.
        root (bool, optional): _description_. Defaults to True.
        rand_direction (bool, optional): _description_. Defaults to False.
        make_vid (bool, optional): _description_. Defaults to False.
    """
    
    img = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    img = img / 255  # put pixels in range [0, 1]
    img = torch.unsqueeze(img, dim=0).float()  # shape (1, 3, 512, 512)

    utils.save_image(img.float().squeeze(), f"Images/{title}/dream_iter{0:>04d}.png")

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
    print("Got freq normed jacobian")

    # get non regularized image embedding
    standard_embed_func = get_embedding_function(freq_reg=False, num_cutouts=num_cutouts)
    image_embedding = standard_embed_func(img)  # shape (512)

    # get tangent vector to image_embedding
    if rand_direction: # get random tangent vector
        v_rand = torch.rand(512).half().cuda()
        v_tang = v_rand - (image_embedding @ v_rand) * image_embedding
        v_tang = v_tang / (v_tang.norm(dim=-1, keepdim=True))

    else: # get singular vector
        # singular value decomposition
        U, S, Vt = jacob_svd(title=title,
                             jacob=jacob,
                             sv_indices=[sv_index])

        v_tang = U[:, sv_index].cuda()  # shape (512)

    # create dream in forward direction (direction of v_tang)
    data_list = [] # target #, step #, step loss, pixel max, pixel min, gradient norm
    clip_dream_frames1 = []
    tgt_not_reached = 0
    tgt_num = 1
    img_dir1 = img.clone()
    
    print("Starting optimization")
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
                                                                    num_cutouts=num_cutouts,
                                                                    root=root)

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
        utils.save_image(img_clipped.float().squeeze(), f"Images/{title}/dream_iter{tgt_num:>04d}.png")
        
    # create dream in backward direction (direction of -v_tang)
    clip_dream_frames2 = []
    tgt_not_reached = 0
    tgt_num = -1
    img_dir2 = img.clone()
    
    print("Starting optimization backward")
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
                                                                    num_cutouts=num_cutouts,
                                                                    root=root)

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
        utils.save_image(img_clipped.float().squeeze(), f"Images/{title}/dream_iter_neg{-tgt_num:>04d}.png")
        
    # target #, step #, step loss, pixel max, pixel min, gradient norm
    df = pandas.DataFrame(data_list, columns=["Target #", "Step #", "Step loss", "Pixel max", "Pixel min", "Gradient norm"])
    df.to_csv(f"Images/{title}/experiment_data.csv")
    
    clip_dream_frames2.reverse()
    
    clip_dream_video = clip_dream_frames2 + [img.permute(0, 2, 3, 1)] + clip_dream_frames1
    clip_dream_video = torch.cat(clip_dream_video, dim=0) # shape (N, H, W, C)
    io.write_video(filename=f"Images/{title}/clip_dream_vid.mp4")

    # # create video
    # if make_vid:
    #     make_video(frames_path=title)
              

def push_img_to_embedding(img, 
                          tgt_embed, 
                          tgt_num, 
                          data_list, 
                          save_every=10, 
                          threshold=0.1, 
                          lr=1., 
                          max_iters=500, 
                          num_cutouts=32, 
                          root=True):

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
        if root:
            loss = torch.norm(input=embed - tgt_embed, p='fro')
        else:
            loss = torch.norm(input=embed - tgt_embed, p='fro') ** 2

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


def make_video(frames_path, fps=5):
    os.system(f"ffmpeg -framerate {fps} -i ./Images/{frames_path}/dream_iter%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./Images/{frames_path}/clip_dream.mp4")


if __name__ == "__main__":
    print("blah")

    clip_dream(img_fp="Images/bird.jpg",
               title="clip_dream2/exp6",
               num_dream_steps=25,
               theta=1,
               sv_index=2,
               threshold=0.1,  # previously was 0.002
               eps=1e-6,
               num_cutouts=32,
               freq_reg='norm',
               lr=250,
               max_iters=1000,
               root=True,
               rand_direction=False,
               show_result=False,
               make_vid=True,
               show_saliency=True,
               print_losses=False,
               print_gradients=False)
import numpy as np
from PIL import Image
import os
import time
import math
import torch as T

import torch
import torch.nn.functional as F_nn

import torchvision.transforms as transforms


def loader(imsize): return transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name, img_size, device):
    image = Image.open(image_name)
    image = loader(img_size)(image).unsqueeze(0)
    image = image.to(device, T.float)
    return image


def warp(x, flo, DEVICE):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F_nn.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F_nn.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output, mask


def flow_setup(args, img_size, k, zfill_length, device, init_path, skip=1):
    flow12 = torch.from_numpy(
        np.load(
            os.path.join(
                args.output_dir,
                f"flow_{img_size}_skip{skip}/flow12_{str(k).zfill(zfill_length)}.npy"))).to(device)
    flow21 = torch.from_numpy(
        np.load(
            os.path.join(
                args.output_dir,
                f"flow_{img_size}_skip{skip}/flow21_{str(k).zfill(zfill_length)}.npy"))).to(device)
    occ_flow21 = torch.from_numpy(
        np.load(
            os.path.join(
                args.output_dir,
                f"flow_{img_size}_skip{skip}/occ_flow21_forward_{str(k).zfill(zfill_length)}.npy"))).to(device)
    occ_flow12 = torch.from_numpy(
        np.load(
            os.path.join(
                args.output_dir,
                f"flow_{img_size}_skip{skip}/occ_flow12_backward_{str(k).zfill(zfill_length)}.npy"))).to(device)
    init_stroke = image_loader(init_path, img_size, device) * 255.
    return flow12, flow21, occ_flow12, occ_flow21, init_stroke


def calc_warping_loss(init_stroke, now_stroke, flow12,
                      flow21, occ_flow12, occ_flow21, device, criterion):
    warped_images_next, warped_mask_next = warp(init_stroke, flow21, device)
    new_mask = 1.0 - ((1.0 - warped_mask_next) + occ_flow21).clamp(0, 1)
    warped_loss = criterion(
        warped_images_next * new_mask / 255.,
        now_stroke * new_mask / 255.)

    warped_images_next, warped_mask_next = warp(now_stroke, flow12, device)
    new_mask = 1.0 - ((1.0 - warped_mask_next) + occ_flow12).clamp(0, 1)
    warped_loss += criterion(warped_images_next *
                             new_mask / 255., init_stroke * new_mask / 255.)

    return warped_loss / 2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

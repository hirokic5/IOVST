import numpy as np
from PIL import Image
import os
import time
import math
import torch as T

import torch
import torch.nn.functional as F_nn

from tqdm import tqdm as tqdm
from pathlib import Path
import cv2
from torchvision.transforms import functional as F
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


def flow_setup(output_dir, origin_size, img_size, k,
               zfill_length, device, init_path, skip=1):
    flow12 = torch.from_numpy(
        np.load(
            os.path.join(
                output_dir,
                f"flow_{img_size}_skip{skip}/flow12_{str(k).zfill(zfill_length)}.npy"))).to(device)
    flow21 = torch.from_numpy(
        np.load(
            os.path.join(
                output_dir,
                f"flow_{img_size}_skip{skip}/flow21_{str(k).zfill(zfill_length)}.npy"))).to(device)
    occ_flow21 = torch.from_numpy(
        np.load(
            os.path.join(
                output_dir,
                f"flow_{img_size}_skip{skip}/occ_flow21_forward_{str(k).zfill(zfill_length)}.npy"))).to(device)
    occ_flow12 = torch.from_numpy(
        np.load(
            os.path.join(
                output_dir,
                f"flow_{img_size}_skip{skip}/occ_flow12_backward_{str(k).zfill(zfill_length)}.npy"))).to(device)
    init_stroke = image_loader(init_path, origin_size, device) * 255.
    if origin_size != img_size:
        flow12 = F.resize(flow12, origin_size)
        flow21 = F.resize(flow21, origin_size)
        occ_flow12 = F.resize(occ_flow12, origin_size)
        occ_flow21 = F.resize(occ_flow21, origin_size)

    return flow12, flow21, occ_flow12, occ_flow21, init_stroke


def calc_warping_loss(init_stroke, now_stroke, flow12,
                      flow21, occ_flow12, occ_flow21, device, criterion, 
                      reduction, window_size=None, pre_mask_12=None, pre_mask_21=None):
    warped_images_next_21, warped_mask_next = warp(init_stroke, flow21, device)
    warped_images_next_21 = warped_images_next_21.clamp(0, 255)
    new_mask_21 = 1.0 - ((1.0 - warped_mask_next) + occ_flow21).clamp(0, 1)
    if pre_mask_21 is not None:
        new_mask_21 = (new_mask_21 - pre_mask_21).clamp(0,1)
    if window_size is None:
        warped_loss = criterion(
            warped_images_next_21 * new_mask_21 / 255.,
            now_stroke * new_mask_21 / 255., reduction=reduction)
    else:
        warped_loss = criterion(
            warped_images_next_21 * new_mask_21 / 255.,
            now_stroke * new_mask_21 / 255., window_size=window_size, reduction=reduction)
        
    warped_images_next_12, warped_mask_next = warp(now_stroke, flow12, device)
    warped_images_next_12 = warped_images_next_12.clamp(0,255)
    new_mask_12 = 1.0 - ((1.0 - warped_mask_next) + occ_flow12).clamp(0, 1)
    if pre_mask_12 is not None:
        new_mask_12 = (new_mask_12 - pre_mask_12).clamp(0,1)
    if window_size is None:
        warped_loss += criterion(warped_images_next_12 *
                                 new_mask_12 / 255., init_stroke * new_mask_12 / 255., reduction=reduction)
    else:
        warped_loss += criterion(warped_images_next_12 *
                                 new_mask_12 / 255., init_stroke * new_mask_12 / 255., window_size=window_size, reduction=reduction)
        
    return warped_loss / 2, new_mask_12, new_mask_21, warped_images_next_12, warped_images_next_21

def histogram_loss(gt,pd,device,method = cv2.HISTCMP_CHISQR):
    hists = []
    for img in [gt,pd]:
        image = img.detach().cpu().numpy()[0].transpose(1,2,0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hist0 = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256])
        hist0 = cv2.normalize(hist0, hist0).flatten()
        hists.append(hist0)
    loss = cv2.compareHist(hists[0], hists[1], method)
    return torch.Tensor([loss]).to(device)

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


def output2video(input_dir, style_path, names, roots, save_dir, c_name,
                 fps=None, zfill_length=3, start=0, end=60, save_name="finaloutput"):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 10 if fps is None else fps
    style = cv2.imread(style_path)
    h, w, c = style.shape
    input_path = os.path.join(
        input_dir, f"{str(start).zfill(zfill_length)}.jpg")
    frame = cv2.imread(input_path)
    height, width, _ = frame.shape
    s_h = min(int(h * (width // 3 / w)), height * 2 // 3)
    thumb_s = cv2.resize(style, (width // 3, s_h))

    video_name = "{}_{}_{}.mp4".format(
        Path(input_dir).stem, Path(style_path).stem, c_name)
    save_path = os.path.join(save_dir, video_name)
    style_name = f"{Path(style_path).stem}"
    for index_frame in tqdm(range(start, end)):
        count = 0
        for name, root in zip(names, roots):
            path = os.path.join(
                root, f"{save_name}_{str(index_frame).zfill(zfill_length)}-{style_name}.jpg")

            if os.path.exists(path):
                count += 1
        if count < len(roots):
            break

    all_frames = index_frame
    print(fps, width, height, all_frames)

    for index_frame in tqdm(range(start, end)):
        outputs = []
        input_path = os.path.join(
            input_dir, f"{str(index_frame).zfill(zfill_length)}.jpg")
        frame = cv2.imread(input_path)
        for name, root in zip(names, roots):
            thumb_c = cv2.resize(frame, (width // 3, height // 3))
            s_h = min(int(h * (width // 3 / w)), height * 2 // 3)
            thumb_s = cv2.resize(style, (width // 3, s_h))
            thumb = np.vstack([thumb_c, thumb_s]) if s_h == height * 2 // 3 else np.vstack(
                [np.zeros((height * 2 // 3 - s_h, width // 3, 3), dtype=np.uint8), thumb_c, thumb_s])
            h_t, w_t, _ = thumb.shape
            thumb = cv2.resize(thumb, (w_t, height))
            path = os.path.join(
                root, f"{save_name}_{str(index_frame).zfill(zfill_length)}-{style_name}.jpg")
            output = cv2.imread(path)
            assert os.path.exists(path), f"path wrong...{path}"
            output = cv2.resize(output, (width, height))
            output = np.hstack([thumb, output])
            cv2.putText(output, name, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 4)
            outputs.append(output)

        for k, o in enumerate(outputs):
            if k == 0:
                output_all = o
            else:
                output_all = np.vstack([output_all, o])
        output_all = cv2.resize(output_all, (width, int(
            output_all.shape[0] * width / output_all.shape[1])))
        if index_frame == start:
            h_all, w_all, _ = output_all.shape
            video = cv2.VideoWriter(save_path, fourcc, fps, (w_all, h_all))

        video.write(output_all)
    video.release()


def after_pad(image, pad=0):
    if pad <= 0:
        return image
    else:
        image[:pad, :, :] = image[pad:pad * 2, :, :]
        image[:, :pad, :] = image[:, pad:pad * 2, :]
        image[-pad:, :, :] = image[-pad * 2:-pad, :, :]
        image[:, -pad:, :] = image[:, -pad * 2:-pad, :]

        return image

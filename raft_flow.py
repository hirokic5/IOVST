from tqdm import tqdm as tqdm
import os
import cv2
import torch
import kornia
import numpy as np
import utils_func

from utils_func import warp
from flow_viz import flow_to_image
from raft_wrapper.utils.utils import InputPadder, coords_grid, bilinear_sampler
from raft_wrapper.raft import RAFT


class RAFT_Flow:
    def __init__(self, raft_args, device):
        self.device = device
        self.model = torch.nn.DataParallel(RAFT(raft_args))
        self.model.load_state_dict(torch.load(raft_args.model))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, start, end, root_dir, output_dir,
                 imgsize, skip=1, zfill_length=5):
        save_dir = os.path.join(output_dir, f"flow_{imgsize}_skip{skip}")
        count = 0
        video = None
        os.makedirs(save_dir, exist_ok=True)
        
        for k in tqdm(range(start, end)):
            # device
            device = torch.device('cuda')
            kernel = torch.ones(1,1).to(device)
            
            if (k - start) > skip - 1:
                image0 = utils_func.image_loader(
                    f"{root_dir}/{str(k-skip).zfill(zfill_length)}.jpg", imgsize, device) * 255.
                image1 = utils_func.image_loader(
                    f"{root_dir}/{str(k).zfill(zfill_length)}.jpg", imgsize, device) * 255.
                padder = InputPadder(image1.shape)
                image0, image1 = padder.pad(image0, image1)

                thresh = 1.0
                occ_thresh = 0.1 - 0.05 * skip / 32
                with torch.no_grad():
                    _, flow12 = self.model(
                        image0, image1, iters=24, test_mode=True)
                    _, flow21 = self.model(
                        image1, image0, iters=24, test_mode=True)

                    coords0 = coords_grid(
                        1, image1.shape[2], image1.shape[3], device)
                    coords1 = coords0 + flow21
                    coords2 = coords1 + \
                        bilinear_sampler(flow12, coords1.permute(0, 2, 3, 1))

                    err = (coords0 - coords2).norm(dim=1)
                    occ_flow21 = (
                        err[0] > thresh).float().unsqueeze(0).unsqueeze(0)

                    warped_images_next_21, warped_mask_next = warp(
                        image0, flow21, device)
                    new_mask_21 = 1.0 - \
                        ((1.0 - warped_mask_next) + occ_flow21).clamp(0, 1)
                    occ21 = (
                        (((new_mask_21 *
                           warped_images_next_21) -
                          image1) /
                         255).norm(
                            dim=1) > occ_thresh).float().unsqueeze(0)
                    occ21 = kornia.morphology.opening(occ21, kernel)
                    occ21 = occ_flow21
                    save_path = os.path.join(
                        save_dir, f"occ_flow21_forward_{str(k).zfill(zfill_length)}.npy")
                    np.save(save_path, occ21.detach().cpu().numpy())

                    coords0 = coords_grid(
                        1, image0.shape[2], image0.shape[3], device)
                    coords1 = coords0 + flow12
                    coords2 = coords1 + \
                        bilinear_sampler(flow21, coords1.permute(0, 2, 3, 1))

                    err = (coords0 - coords2).norm(dim=1)
                    occ_flow12 = (
                        err[0] > thresh).float().unsqueeze(0).unsqueeze(0)

                    warped_images_next_12, warped_mask_next = warp(
                        image1, flow12, device)
                    new_mask_12 = 1.0 - \
                        ((1.0 - warped_mask_next) + occ_flow12).clamp(0, 1)
                    occ12 = (
                        (((new_mask_12 *
                           warped_images_next_12) -
                          image0) /
                         255).norm(
                            dim=1) > occ_thresh).float().unsqueeze(0)
                    occ12 = kornia.morphology.opening(occ12, kernel)
                    occ12 = occ_flow12
                    save_path = os.path.join(
                        save_dir, f"occ_flow12_backward_{str(k).zfill(zfill_length)}.npy")
                    np.save(save_path, occ12.detach().cpu().numpy())

                view_all = np.vstack(
                    [
                        np.hstack(
                            [
                                image0.detach().cpu().numpy()[
                                    0].transpose(1, 2, 0),
                                image1.detach().cpu().numpy()[
                                    0].transpose(1, 2, 0),
                            ]
                        ),
                        np.hstack(
                            [
                                flow_to_image(flow21.detach().cpu().numpy()[
                                              0].transpose(1, 2, 0)),
                                flow_to_image(flow12.detach().cpu().numpy()[
                                              0].transpose(1, 2, 0)),
                            ]
                        ),
                        np.hstack(
                            [
                                cv2.cvtColor(occ21.detach().cpu().numpy()[0,0] * 255,cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(occ12.detach().cpu().numpy()[0,0] * 255,cv2.COLOR_GRAY2BGR),
                            ]
                        ),
                    ]
                ).astype(np.uint8)
                view_all = cv2.cvtColor(view_all, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(
                    save_dir, f"view_all_{str(k).zfill(zfill_length)}.jpg")
                cv2.imwrite(save_path, view_all)
                if count == 0:
                    fps = 5
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    video_path = os.path.join(save_dir, "view_all_video.mp4")
                    h_all, w_all, _ = view_all.shape
                    video = cv2.VideoWriter(
                        video_path, fourcc, fps, (w_all, h_all))

                save_path = os.path.join(
                    save_dir, f"flow12_{str(k).zfill(zfill_length)}.npy")
                np.save(save_path, flow12.detach().cpu().numpy())

                save_path = os.path.join(
                    save_dir, f"flow21_{str(k).zfill(zfill_length)}.npy")
                np.save(save_path, flow21.detach().cpu().numpy())

                video.write(view_all)
                count += 1

        if video is not None:
            video.release()

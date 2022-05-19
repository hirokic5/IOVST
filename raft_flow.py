from tqdm import tqdm as tqdm
import os
import torch
import numpy as np
import utils_func

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
        os.makedirs(save_dir, exist_ok=True)
        for k in tqdm(range(start, end + 1)):
            # device
            device = torch.device('cuda')

            if (k - start) > skip - 1:
                image0 = utils_func.image_loader(
                    f"{root_dir}/{str(k-skip).zfill(zfill_length)}.jpg", imgsize, device) * 255.
                image1 = utils_func.image_loader(
                    f"{root_dir}/{str(k).zfill(zfill_length)}.jpg", imgsize, device) * 255.
                padder = InputPadder(image1.shape)
                image0, image1 = padder.pad(image0, image1)
                
                thresh = 1.0
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
                    occ = (err[0] > thresh).float().unsqueeze(0).unsqueeze(0)

                    save_path = os.path.join(
                        save_dir, f"occ_flow21_forward_{str(k).zfill(zfill_length)}.npy")
                    np.save(save_path, occ.detach().cpu().numpy())

                    coords0 = coords_grid(
                        1, image0.shape[2], image0.shape[3], device)
                    coords1 = coords0 + flow12
                    coords2 = coords1 + \
                        bilinear_sampler(flow21, coords1.permute(0, 2, 3, 1))

                    err = (coords0 - coords2).norm(dim=1)
                    occ = (err[0] > thresh).float().unsqueeze(0).unsqueeze(0)

                    save_path = os.path.join(
                        save_dir, f"occ_flow12_backward_{str(k).zfill(zfill_length)}.npy")
                    np.save(save_path, occ.detach().cpu().numpy())

                save_path = os.path.join(
                    save_dir, f"flow12_{str(k).zfill(zfill_length)}.npy")
                np.save(save_path, flow12.detach().cpu().numpy())

                save_path = os.path.join(
                    save_dir, f"flow21_{str(k).zfill(zfill_length)}.npy")
                np.save(save_path, flow21.detach().cpu().numpy())

import os
import torch
import torch as T
import torch.optim as optim
from torchvision.transforms import functional as F
import torch.nn.functional as F_nn
import argparse
import numpy as np
from param_stroke import BrushStrokeRenderer
import losses
import datetime
import cv2
from importlib import import_module
import shutil
from tqdm import tqdm as tqdm

from RAFT.core.utils.utils import InputPadder
from raft_flow import RAFT_Flow
from utils_func import init_logger, image_loader, flow_setup, calc_warping_loss


def main(args, config_path):
    ########################
    # prepare data & setting
    ########################

    if args.input_video is not None:
        cap = cv2.VideoCapture(args.input_video)
        all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(args.root_dir, exist_ok=True)
        zfill_length = len(str(all_frames))
        for index_frame in tqdm(range(args.start_frame, args.end_frame + 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame)
            ret, frame = cap.read()
            if ret:
                save_path = os.path.join(
                    args.root_dir, f"{str(index_frame).zfill(zfill_length)}.jpg")
                cv2.imwrite(save_path, frame)
    else:
        zfill_length = args.zfill_length

    device = torch.device(args.device)
    raft = RAFT_Flow(args.raft_args, device=device)
    for imgsize in [args.stroke_img_size, args.pixel_size]:
        for skip in [1, 2]:
            raft(
                args.start_frame,
                args.end_frame,
                args.root_dir,
                args.output_dir,
                imgsize,
                skip=skip,
                zfill_length=zfill_length)

    model_name = args.model_name
    root = os.path.join(args.output_dir, model_name, "{}".format(
        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    os.makedirs(root, exist_ok=True)

    log_path = os.path.join(root, "train.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    LOGGER = init_logger(log_path)

    shutil.copy(f"{config_path}.py", root)

    start_whole = datetime.datetime.now()

    if args.criterion == "l1loss":
        criterion = F_nn.l1_loss
    elif args.criterion == "mseloss":
        criterion = F_nn.mse_loss

    ####################
    # styletransfer loop
    ####################

    for u, k in enumerate(range(args.start_frame, args.end_frame)):
        style_img_file = args.style_img_file
        content_img_file = f"{args.root_dir}/{str(k).zfill(zfill_length)}.jpg"

        start = datetime.datetime.now()
        LOGGER.info(f'style:{style_img_file}, content:{content_img_file}')

        vgg_weight_file = args.vgg_weight_file
        print_freq = 10

        imsize = args.stroke_img_size
        content_img = image_loader(content_img_file, imsize, device)
        style_img = image_loader(style_img_file, imsize, device)
        output_name = f'finaloutput_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}'

        bs_content_layers = ['conv4_1', 'conv5_1']
        bs_style_layers = [
            'conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1']
        px_content_layers = [
            'conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1']
        px_style_layers = [
            'conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1']

        #################################
        # optimize the canvas brushstroke
        #################################
        if args.brushstroke:
            # brush strokes parameters
            canvas_color = args.canvas_color
            num_strokes = args.num_strokes
            samples_per_curve = args.samples_per_curve
            brushes_per_pixel = args.brushes_per_pixel
            _, _, H, W = content_img.shape
            canvas_height = H
            canvas_width = W
            length_scale = 1.1
            width_scale = 0.1

            vgg_loss = losses.StyleTransferLosses(vgg_weight_file, content_img, style_img,
                                                  bs_content_layers, bs_style_layers, scale_by_y=True)
            vgg_loss.to(device).eval()

            num_steps = 2 if args.debug else args.stroke_steps + args.stroke_steps_warp
            style_weight = args.stroke_style_weight  # 10000.
            content_weight = args.stroke_content_weight  # 1.
            tv_weight = args.stroke_tv_weight  # 1.#0.008#tv_weight=0
            curv_weight = args.stroke_curv_weight
            warp_weight = args.stroke_warp_weight  # 1.#0.008#tv_weight=0
            skip_warp_weight = args.stroke_skip_warp_weight

            if args.brush_initialize or u == 0:
                bs_renderer = BrushStrokeRenderer(canvas_height, canvas_width, num_strokes, samples_per_curve, brushes_per_pixel,
                                                  canvas_color, length_scale, width_scale,
                                                  content_img=content_img[0].permute(1, 2, 0).cpu().numpy())
            bs_renderer.to(device)

            optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                    bs_renderer.curve_c, bs_renderer.width], lr=1e-1)
            optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

            LOGGER.info('Optimizing brushstroke-styled canvas..')

            if k > 0 and u > 0:
                init_path = os.path.join(
                    root, f'stroke_final_{str(k-1).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                )
                flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                    args, args.stroke_img_size, k, zfill_length, device, init_path, skip=1)
            if args.skip_warp and k > 1 and u > 1:
                init_path = os.path.join(
                    root, f'stroke_stylized_{str(k-2).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                )
                flow12_skip, flow21_skip, occ_flow12_skip, occ_flow21_skip, init_stroke_skip = flow_setup(
                    args, args.stroke_img_size, k, zfill_length, device, init_path, skip=2)

            for iteration in range(num_steps):
                if iteration == args.stroke_steps and k > 0 and u > 0:
                    style_weight = args.stroke_style_weight_warp  # 10000.
                    content_weight = args.stroke_content_weight_warp  # 1.
                    optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                            bs_renderer.curve_c, bs_renderer.width], lr=1e-2)
                    optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

                optimizer.zero_grad()
                optimizer_color.zero_grad()
                input_img = bs_renderer()
                input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
                b, c, origin_h, origin_w = input_img.shape
                content_score, style_score = vgg_loss(input_img)

                style_score *= style_weight
                content_score *= content_weight
                tv_score = tv_weight * losses.total_variation_loss(bs_renderer.location, bs_renderer.curve_s,
                                                                   bs_renderer.curve_e, K=10)
                curv_score = curv_weight * \
                    losses.curvature_loss(
                        bs_renderer.curve_s,
                        bs_renderer.curve_e,
                        bs_renderer.curve_c)
                loss = style_score + content_score + tv_score + curv_score
                color_loss = style_score.clone()

                ###############
                # warping score
                ###############
                if k > 0 and u > 0:
                    now_stroke = input_img * 255.
                    if iteration == 0:
                        padder = InputPadder(init_stroke.shape)
                        init_stroke, now_stroke = padder.pad(
                            init_stroke, now_stroke)
                    else:
                        padder = InputPadder(now_stroke.shape)
                        _, now_stroke = padder.pad(init_stroke, now_stroke)

                    warped_loss = calc_warping_loss(
                        init_stroke,
                        now_stroke,
                        flow12,
                        flow21,
                        occ_flow12,
                        occ_flow21,
                        device,
                        criterion) * warp_weight

                    loss += warped_loss
                    color_loss += warped_loss
                else:
                    warped_loss = torch.Tensor([0])

                if args.skip_warp and k > 1 and u > 1:
                    now_stroke = input_img * 255.
                    if iteration == 0:
                        padder = InputPadder(init_stroke_skip.shape)
                        init_stroke_skip, now_stroke = padder.pad(
                            init_stroke_skip, now_stroke)
                    else:
                        padder = InputPadder(now_stroke.shape)
                        _, now_stroke = padder.pad(
                            init_stroke_skip, now_stroke)

                    warped_loss_skip = calc_warping_loss(
                        init_stroke_skip,
                        now_stroke,
                        flow12_skip,
                        flow21_skip,
                        occ_flow12_skip,
                        occ_flow21_skip,
                        device,
                        criterion) * skip_warp_weight

                    loss += warped_loss_skip
                    color_loss += warped_loss_skip
                    warped_loss += warped_loss_skip

                loss.backward(inputs=[bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                      bs_renderer.curve_c, bs_renderer.width], retain_graph=True)
                optimizer.step()

                color_loss.backward(inputs=[bs_renderer.color])
                optimizer_color.step()

                #########
                # logging
                #########
                if iteration % print_freq == 0:
                    LOGGER.info(f'[{iteration}/{num_steps}] stroke, style loss:{style_score.item():.3f}, content loss:{content_score.item():.3f}, tv loss:{tv_score.item():.3f}, curvature loss:{curv_score.item():.3f}, warped loss:{warped_loss.item():.3f} ')
                    if iteration >= args.stroke_steps and k > 0 and u > 0:
                        save_path = os.path.join(
                            root, f'stroke_stylized_warped_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                    else:
                        save_path = os.path.join(
                            root, f'stroke_stylized_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                    save = (
                        input_img[0].detach().cpu().numpy().transpose(
                            1,
                            2,
                            0) *
                        255.).astype(
                        np.uint8)
                    save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, save)

                    if iteration == 0:
                        save_path = os.path.join(
                            root, f'stroke_init_stylized_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                        save = (
                            input_img[0].detach().cpu().numpy().transpose(
                                1,
                                2,
                                0) *
                            255.).astype(
                            np.uint8)
                        save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, save)

            with T.no_grad():
                input_img = bs_renderer()

            input_img = input_img.detach()[None].permute(
                0, 3, 1, 2).contiguous()
            save_path = os.path.join(
                root, f'stroke_final_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
            )
            save = (
                input_img[0].detach().cpu().numpy().transpose(
                    1,
                    2,
                    0) *
                255.).astype(
                np.uint8)
            save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save)
            model_path = os.path.join(root, f'renderer_{str(k).zfill(4)}.pth')
            torch.save(bs_renderer.state_dict(), model_path)

        else:
            if args.initial_input is None:
                input_img = content_img.clone()
            else:
                initial_img_file = os.path.join(
                    args.initial_input, f"{output_name}.jpg")
                input_img = image_loader(initial_img_file, imsize, device)

        ################################
        # optimize the canvas pixel-wise
        ################################
        print_freq = 100
        if args.debug:
            num_steps = 10
        else:
            num_steps = args.pixel_steps + \
                args.pixel_steps_warp if k > 0 and u > 0 else args.pixel_steps
        style_weight = args.pixel_style_weight  # 10000.
        content_weight = args.pixel_content_weight  # 1.
        tv_weight = args.pixel_tv_weight  # 1.#0.008#tv_weight=0
        warp_weight = args.pixel_warp_weight  # 1.#0.008#tv_weight=0
        skip_warp_weight = args.pixel_skip_warp_weight
        resize_size = args.pixel_size

        content_img_resized = F.resize(content_img, resize_size)
        style_img_resized = F.resize(style_img, resize_size)
        print(torch.min(input_img), torch.max(input_img))
        input_img = F.resize(input_img, resize_size)
        input_img = T.nn.Parameter(input_img, requires_grad=True)

        vgg_loss = losses.StyleTransferLosses(vgg_weight_file, content_img_resized, style_img_resized,
                                              px_content_layers, px_style_layers)
        vgg_loss.to(device).eval()
        optimizer = optim.Adam([input_img], lr=1e-3)
        LOGGER.info('Optimizing pixel-wise canvas..')

        if k > 0 and u > 0:
            init_path = os.path.join(
                root, f'finaloutput_{str(k-1).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
            )
            flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                args, args.pixel_size, k, zfill_length, device, init_path, skip=1)
        if args.skip_warp and k > 1 and u > 1:
            init_path = os.path.join(
                root, f'finaloutput_{str(k-2).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
            )
            flow12_skip, flow21_skip, occ_flow12_skip, occ_flow21_skip, init_stroke_skip = flow_setup(
                args, args.pixel_size, k, zfill_length, device, init_path, skip=2)

        for iteration in range(num_steps):
            if iteration == args.pixel_steps and k > 0 and u > 0:
                optimizer = optim.Adam([input_img], lr=1e-3)
                style_weight = args.pixel_style_weight_warp  # 10000.
                content_weight = args.pixel_content_weight_warp  # 1.

            optimizer.zero_grad()
            input = T.clamp(input_img, 0., 1.)
            content_score, style_score = vgg_loss(input)

            style_score *= style_weight
            content_score *= content_weight
            tv_score = tv_weight * losses.tv_loss(input)
            loss = style_score + content_score + tv_score

            ###############
            # warping score
            ###############
            if k > 0 and u > 0:

                now_stroke = input_img * 255.
                if iteration == 0:
                    padder = InputPadder(init_stroke.shape)
                    init_stroke, now_stroke = padder.pad(
                        init_stroke, now_stroke)
                else:
                    padder = InputPadder(now_stroke.shape)
                    _, now_stroke = padder.pad(init_stroke, now_stroke)

                warped_loss = calc_warping_loss(
                    init_stroke,
                    now_stroke,
                    flow12,
                    flow21,
                    occ_flow12,
                    occ_flow21,
                    device,
                    criterion) * warp_weight

                loss += warped_loss
            else:
                warped_loss = torch.Tensor([0])

            if args.skip_warp and k > 1 and u > 1:
                now_stroke = input_img * 255.
                if iteration == 0:
                    padder = InputPadder(init_stroke_skip.shape)
                    init_stroke_skip, now_stroke = padder.pad(
                        init_stroke_skip, now_stroke)
                else:
                    padder = InputPadder(now_stroke.shape)
                    _, now_stroke = padder.pad(init_stroke_skip, now_stroke)

                warped_loss_skip = calc_warping_loss(
                    init_stroke_skip,
                    now_stroke,
                    flow12_skip,
                    flow21_skip,
                    occ_flow12_skip,
                    occ_flow21_skip,
                    device,
                    criterion) * skip_warp_weight

                loss += warped_loss_skip
                warped_loss += warped_loss_skip

            loss.backward(inputs=[input_img])
            optimizer.step()

            #########
            # logging
            #########
            if iteration % print_freq == 0:
                LOGGER.info(
                    f'[{iteration}/{num_steps}] pixel, style loss:{style_score:.3f}, content loss:{content_score:.3f}, tv loss:{tv_score:.3f}, warped loss:{warped_loss.item():.3f}')
                if iteration >= args.pixel_steps and k > 0 and u > 0:
                    save_path = os.path.join(
                        root, f'pixel_stylized_warped_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                    )
                else:
                    save_path = os.path.join(
                        root, f'pixel_stylized_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                    )
                save = (
                    input[0].detach().cpu().numpy().transpose(
                        1,
                        2,
                        0) *
                    255.).astype(
                    np.uint8)
                save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save)

        output = T.clamp(input_img, 0., 1.)

        save_path = os.path.join(
            root, f'finaloutput_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
        )

        save = (
            output[0].detach().cpu().numpy().transpose(
                1,
                2,
                0) *
            255.).astype(
            np.uint8)
        save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save)

        LOGGER.info('Finished!')
        LOGGER.info(f'elapsed time:{datetime.datetime.now()-start}')

    LOGGER.info('Finished! ALL !')
    LOGGER.info(f'elapsed time:{datetime.datetime.now()-start_whole}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, default="setting")
    args = parser.parse_args()
    params = import_module(args.params)
    main(params.CFG, args.params)

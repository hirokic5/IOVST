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

from raft_wrapper.utils.utils import InputPadder
from raft_flow import RAFT_Flow
from utils_func import init_logger, image_loader, warp, flow_setup, calc_warping_loss, output2video, after_pad
from utils_brush import gram_matrix

def main(args, config_path):
    ########################
    # prepare data & setting
    ########################

    if args.input_video is not None:
        assert os.path.exists(
            args.input_video), "the path to video seems to be wrong..."
        cap = cv2.VideoCapture(args.input_video)
        all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(args.root_dir, exist_ok=True)
        zfill_length = len(str(all_frames))
        prev_frame = None
        counts = 0
        for index_frame in tqdm(range(args.start_frame, all_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame)
            ret, frame = cap.read()
            if ret:
                if prev_frame is not None:
                    diff = np.sum(cv2.absdiff(frame,prev_frame)) / frame.flatten().shape[0] > 1.0
                else:
                    diff = True
                prev_frame = frame
                if diff:
                    save_path = os.path.join(
                        args.root_dir, f"{str(args.start_frame + counts).zfill(zfill_length)}.jpg")
                    cv2.imwrite(save_path, frame)
                    counts += 1
                    
            if counts >= args.end_frame - args.start_frame:
                break

                
                
        cap.release()
    else:
        zfill_length = args.zfill_length

    device = torch.device(args.device)
    if args.warping_scheme is not None:
        raft = RAFT_Flow(args.raft_args, device=device)
        for imgsize in [args.stroke_img_size, args.pixel_size]:
            for skip in args.warping_scheme:
                raft(
                    args.start_frame,
                    args.end_frame,
                    args.root_dir,
                    args.output_dir,
                    imgsize,
                    skip=skip,
                    zfill_length=zfill_length)

    model_name = args.model_name
    if args.brushstroke:
        model_name += "_Brushstroke"
    else:
        model_name += "_noBrushstroke"
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

    if args.warping_mode_pixel == 0:
        save_name_pixel = "pixel_stylized_forward"
    elif args.warping_mode_pixel == 1:
        save_name_pixel = "finaloutput"
        
    
    
    ####################
    # styletransfer loop
    ####################
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
        for now_frame in range(args.start_frame, args.end_frame):
            style_img_file = args.style_img_file
            content_img_file = f"{args.root_dir}/{str(now_frame).zfill(zfill_length)}.jpg"

            start = datetime.datetime.now()
            LOGGER.info(f'style:{style_img_file}, content:{content_img_file}')

            vgg_weight_file = args.vgg_weight_file
            print_freq = 10

            imsize = args.stroke_img_size
            content_img = image_loader(content_img_file, imsize, device)
            padder = InputPadder(content_img.shape)
            content_img = padder.pad(content_img)[0]
            style_img = image_loader(style_img_file, imsize, device)
        
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

            if args.brush_initialize or now_frame - args.start_frame == 0:
                bs_renderer = BrushStrokeRenderer(canvas_height, canvas_width, num_strokes, samples_per_curve, brushes_per_pixel,
                                                  canvas_color, length_scale, width_scale,
                                                  content_img=content_img[0].permute(1, 2, 0).cpu().numpy())
            bs_renderer.to(device)

            optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                    bs_renderer.curve_c, bs_renderer.width], lr=1e-1)
            optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

            
            scaler = torch.cuda.amp.GradScaler()
            scaler_color = torch.cuda.amp.GradScaler()

            
            LOGGER.info('Optimizing brushstroke-styled canvas..')

            if args.warping_scheme is not None:
                flow_info = {}
                for skip in args.warping_scheme:
                    if now_frame - args.start_frame > skip - 1:
                        init_path = os.path.join(
                            root, f'stroke_forward_{str(now_frame-skip).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                        flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                            args, args.stroke_img_size, now_frame, zfill_length, device, init_path, skip=skip
                        )
                        flow_info[skip] = {
                            "flow12": flow12,
                            "flow21": flow21,
                            "occ_flow12": occ_flow12,
                            "occ_flow21": occ_flow21,
                            "init_stroke": init_stroke,
                        }


            for iteration in range(num_steps):
                if iteration == args.stroke_steps and now_frame - \
                        args.start_frame > 0 and args.warping_scheme is not None:
                    style_weight = args.stroke_style_weight_warp  # 10000.
                    content_weight = args.stroke_content_weight_warp  # 1.
                    optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                            bs_renderer.curve_c, bs_renderer.width], lr=1e-2)
                    optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

                if iteration == args.stroke_steps and now_frame == args.start_frame:
                    break
                    
                optimizer.zero_grad()
                optimizer_color.zero_grad()
                input_img = bs_renderer()
                input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
                b, c, origin_h, origin_w = input_img.shape
                with torch.cuda.amp.autocast(enabled=args.amp):
                    content_score, style_score = vgg_loss(input_img)
                    style_score *= style_weight
                    content_score *= content_weight
                    tv_score = tv_weight * losses.total_variation_loss(bs_renderer.location, bs_renderer.curve_s,
                                                                       bs_renderer.curve_e, K=10)
                    curv_score = curv_weight * \
                        losses.curvature_loss(
                            bs_renderer.curve_s,
                            bs_renderer.curve_e,
                            bs_renderer.curve_c
                    )
                                                                       
                                                                       
                loss = style_score + content_score + tv_score + curv_score
                color_loss = style_score.clone()

                ###############
                # warping score
                ###############
                warped_loss = torch.Tensor([0]).to(device)
                warped_num = 0
                if args.warping_scheme is not None:
                    for skip in args.warping_scheme:
                        if now_frame - args.start_frame > skip - 1:
                            now_stroke = input_img * 255.
                            
                            with torch.cuda.amp.autocast(enabled=args.amp):
                                warped_loss_skip, occlusion_mask, warped_images_next = calc_warping_loss(
                                    flow_info[skip]["init_stroke"],
                                    now_stroke,
                                    flow_info[skip]["flow12"],
                                    flow_info[skip]["flow21"],
                                    flow_info[skip]["occ_flow12"],
                                    flow_info[skip]["occ_flow21"],
                                    device,
                                    criterion,
                                    args.stroke_reduction
                                )
                            
                            warped_loss += warped_loss_skip * warp_weight
                            warped_num += 1

                if warped_num > 0:
                    warped_loss = warped_loss / warped_num
                    loss += warped_loss[0]
                    color_loss += warped_loss[0]

                scaler.scale(loss).backward(inputs=[bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                      bs_renderer.curve_c, bs_renderer.width], retain_graph=True)
                scaler.step(optimizer)
                scaler.update()

                scaler_color.scale(color_loss).backward(inputs=[bs_renderer.color])
                scaler_color.step(optimizer_color)
                scaler_color.update()

                #########
                # logging
                #########
                if iteration % print_freq == 0:
                    LOGGER.info(f'[{iteration}/{num_steps}] stroke, style loss:{style_score.item():.3f}, content loss:{content_score.item():.3f}, tv loss:{tv_score.item():.3f}, curvature loss:{curv_score.item():.3f}, warped loss:{warped_loss.item():.3f} ')
                    
                    save_path = os.path.join(
                        root, f'stroke_forward_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
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
            
            save = (
                input_img[0].detach().cpu().numpy().transpose(
                    1,
                    2,
                    0) *
                255.).astype(
                np.uint8)
            save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save)
            
        stroke_name = "stroke_forward"                                                           
        #################################
        # brushstroke cycle
        #################################                                                             
        if args.warping_mode_stroke >= 1:
                                                                       
            for now_frame in range(args.start_frame, args.end_frame):
                style_img_file = args.style_img_file
                content_img_file = f"{args.root_dir}/{str(now_frame).zfill(zfill_length)}.jpg"

                start = datetime.datetime.now()
                LOGGER.info(f'style:{style_img_file}, content:{content_img_file}')

                vgg_weight_file = args.vgg_weight_file
                print_freq = 10

                imsize = args.stroke_img_size
                content_img = image_loader(content_img_file, imsize, device)
                padder = InputPadder(content_img.shape)
                content_img = padder.pad(content_img)[0]
                style_img = image_loader(style_img_file, imsize, device)
                
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

                if args.brush_initialize or now_frame - args.start_frame == 0:
                    bs_renderer = BrushStrokeRenderer(canvas_height, canvas_width, num_strokes, samples_per_curve, brushes_per_pixel,
                                                      canvas_color, length_scale, width_scale,
                                                      content_img=content_img[0].permute(1, 2, 0).cpu().numpy())
                bs_renderer.to(device)

                optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                        bs_renderer.curve_c, bs_renderer.width], lr=1e-1)
                optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)


                scaler = torch.cuda.amp.GradScaler()
                scaler_color = torch.cuda.amp.GradScaler()


                LOGGER.info('Optimizing brushstroke-styled canvas..')

                if args.warping_scheme is not None:
                    flow_info = {
                        "forward":{},
                        "backward":{}
                    }
                    for skip in args.warping_scheme:
                        if now_frame - args.start_frame > skip - 1:
                            init_path = os.path.join(
                                root, f'stroke_forward_{str(now_frame-skip).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                            )
                            flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                                args, args.stroke_img_size, now_frame, zfill_length, device, init_path, skip=skip
                            )
                            flow_info["forward"][skip] = {
                                "flow12": flow12,
                                "flow21": flow21,
                                "occ_flow12": occ_flow12,
                                "occ_flow21": occ_flow21,
                                "init_stroke": init_stroke,
                            }
                        if args.end_frame - now_frame > skip:
                            init_path = os.path.join(
                                root, f'stroke_forward_{str(now_frame+skip).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                            )
                            flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                                args, args.stroke_img_size, now_frame+skip, zfill_length, device, init_path, skip=skip
                            )
                            flow_info["backward"][skip] = {
                                "flow12": flow12,
                                "flow21": flow21,
                                "occ_flow12": occ_flow12,
                                "occ_flow21": occ_flow21,
                                "init_stroke": init_stroke,
                            }


                for iteration in range(num_steps):
                    if iteration == args.stroke_steps and now_frame - \
                            args.start_frame > 0 and args.warping_scheme is not None:
                        style_weight = args.stroke_style_weight_warp  # 10000.
                        content_weight = args.stroke_content_weight_warp  # 1.
                        optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                                bs_renderer.curve_c, bs_renderer.width], lr=1e-2)
                        optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

                    if iteration == args.stroke_steps and now_frame == args.start_frame:
                        break

                    optimizer.zero_grad()
                    optimizer_color.zero_grad()
                    input_img = bs_renderer()
                    input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
                    b, c, origin_h, origin_w = input_img.shape
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        content_score, style_score = vgg_loss(input_img)
                        style_score *= style_weight
                        content_score *= content_weight
                        tv_score = tv_weight * losses.total_variation_loss(bs_renderer.location, bs_renderer.curve_s,
                                                                       bs_renderer.curve_e, K=10)
                        curv_score = curv_weight * \
                            losses.curvature_loss(
                                bs_renderer.curve_s,
                                bs_renderer.curve_e,
                                bs_renderer.curve_c
                        )



                    loss = style_score + content_score + tv_score + curv_score
                    color_loss = style_score.clone()

                    ###############
                    # warping score
                    ###############
                    warped_loss = torch.Tensor([0]).to(device)
                    warped_num = 0
                    if args.warping_scheme is not None:
                        for skip in args.warping_scheme:
                            if now_frame - args.start_frame > skip - 1:
                                now_stroke = input_img * 255.
                                with torch.cuda.amp.autocast(enabled=args.amp):
                                    warped_loss_skip, occlusion_mask, warped_images_next = calc_warping_loss(
                                        flow_info["forward"][skip]["init_stroke"],
                                        now_stroke,
                                        flow_info["forward"][skip]["flow12"],
                                        flow_info["forward"][skip]["flow21"],
                                        flow_info["forward"][skip]["occ_flow12"],
                                        flow_info["forward"][skip]["occ_flow21"],
                                        device,
                                        criterion,
                                        args.stroke_reduction
                                    )

                                warped_loss += warped_loss_skip * warp_weight
                                warped_num += 1

                            if args.end_frame - now_frame > skip:
                                now_stroke = input_img * 255.
                                with torch.cuda.amp.autocast(enabled=args.amp):
                                    warped_loss_skip, occlusion_mask, warped_images_next = calc_warping_loss(
                                        now_stroke,
                                        flow_info["backward"][skip]["init_stroke"],
                                        flow_info["backward"][skip]["flow12"],
                                        flow_info["backward"][skip]["flow21"],
                                        flow_info["backward"][skip]["occ_flow12"],
                                        flow_info["backward"][skip]["occ_flow21"],
                                        device,
                                        criterion,
                                        args.stroke_reduction
                                    )
                                warped_loss += warped_loss_skip * warp_weight
                                warped_num += 1
                    
                    if warped_num > 0:
                        warped_loss = warped_loss / warped_num
                        loss += warped_loss[0]
                        color_loss += warped_loss[0]

                    scaler.scale(loss).backward(inputs=[bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                                          bs_renderer.curve_c, bs_renderer.width], retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()

                    scaler_color.scale(color_loss).backward(inputs=[bs_renderer.color])
                    scaler_color.step(optimizer_color)
                    scaler_color.update()

                    #########
                    # logging
                    #########
                    if iteration % print_freq == 0:
                        LOGGER.info(f'[{iteration}/{num_steps}] stroke, style loss:{style_score.item():.3f}, content loss:{content_score.item():.3f}, tv loss:{tv_score.item():.3f}, curvature loss:{curv_score.item():.3f}, warped loss:{warped_loss.item():.3f} ')
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

                with T.no_grad():
                    input_img = bs_renderer()

                input_img = input_img.detach()[None].permute(
                    0, 3, 1, 2).contiguous()
                
                save = (
                    input_img[0].detach().cpu().numpy().transpose(
                        1,
                        2,
                        0) *
                    255.).astype(
                    np.uint8)
                save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save)
            stroke_name = "stroke_final"       

    ########################################
    # optimize the canvas pixel-wise forward
    ########################################
                
    for now_frame in range(args.start_frame, args.end_frame):
        style_img_file = args.style_img_file
        content_img_file = f"{args.root_dir}/{str(now_frame).zfill(zfill_length)}.jpg"

        start = datetime.datetime.now()
        LOGGER.info(f'style:{style_img_file}, content:{content_img_file}')

        vgg_weight_file = args.vgg_weight_file
        print_freq = 10

        
        print_freq = 100
        if args.debug:
            num_steps = 10
        else:
            num_steps = args.pixel_steps + args.pixel_steps_warp_1st
            
        style_weight = args.pixel_style_weight  # 10000.
        content_weight = args.pixel_content_weight  # 1.
        tv_weight = args.pixel_tv_weight  # 1.#0.008#tv_weight=0
        warp_weight = args.pixel_warp_weight  # 1.#0.008#tv_weight=0
        resize_size = args.pixel_size

        content_img_resized = image_loader(content_img_file, resize_size, device)
        padder = InputPadder(content_img_resized.shape)
        content_img_resized = padder.pad(content_img_resized)[0]
        style_img_resized = F.resize(style_img, resize_size)
        _,_,h,w = content_img_resized.shape
        if args.brushstroke:
            save_path = os.path.join(
                root, f'{stroke_name}_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
            )
            input_img = image_loader(save_path, resize_size, device)
        else:
            input_img = content_img_resized.clone()
        input_img = T.nn.Parameter(input_img, requires_grad=True)
        print(input_img.shape,content_img.shape,content_img_resized.shape)

        vgg_loss = losses.StyleTransferLosses(vgg_weight_file, content_img_resized, style_img_resized,
                                              px_content_layers, px_style_layers)
        vgg_loss.to(device).eval()
        
        scaler = torch.cuda.amp.GradScaler()
        
        if args.warping_scheme is not None:
            flow_info = {}
            for skip in args.warping_scheme:
                if now_frame - args.start_frame > skip - 1:
                    init_path = os.path.join(
                        root, f'pixel_stylized_forward_{str(now_frame-skip).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                    )
                    flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                        args, resize_size, now_frame, zfill_length, device, init_path, skip=skip
                    )
                    flow_info[skip] = {
                        "flow12": flow12,
                        "flow21": flow21,
                        "occ_flow12": occ_flow12,
                        "occ_flow21": occ_flow21,
                        "init_stroke": init_stroke,
                    }
                    

        
        optimizer = optim.Adam([input_img], lr=1e-3)
        LOGGER.info('Optimizing pixel-wise canvas..')

        
                    
        for iteration in range(num_steps):            
            if iteration == args.pixel_steps and now_frame - \
                args.start_frame > 0 and args.warping_scheme is not None:
                if args.occlusion_replace:
                    input_img_forward = input_img
                    warped_images = None
                    new_mask = None
                    for skip in args.warping_scheme:
                        if now_frame - args.start_frame > skip - 1:
                            flow21 = flow_info[skip]["flow21"]
                            occ_flow21 = flow_info[skip]["occ_flow21"]
                            save_path = os.path.join(
                                root, f'pixel_stylized_forward_{os.path.basename(f"{args.root_dir}/{str(now_frame-skip).zfill(zfill_length)}.jpg").split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                            )
                            pixel_stroke = image_loader(save_path, resize_size, device) * 255.
                            warped_images_next_21, warped_mask_next = warp(pixel_stroke, flow21, device)
                            new_mask_21 = 1.0 - ((1.0 - warped_mask_next) + occ_flow21).clamp(0, 1)
                            input_img_forward = warped_images_next_21 / 255. * new_mask_21 + \
                                input_img * (1 - new_mask_21)
                            if args.occlusion_replace_main == skip:
                                warped_images = warped_images_next_21
                            if new_mask is None:
                                new_mask = new_mask_21
                            else:
                                new_mask = new_mask_21 * new_mask
                    
                    if input_img_forward is not None:
                        input_img = T.nn.Parameter(input_img_forward, requires_grad=True)
                optimizer = optim.Adam([input_img], lr=1e-3)
                warp_weight = args.pixel_warp_weight  # 1.#0.008#tv_weight=0
                style_weight = args.pixel_style_weight_warp  # 10000.
                content_weight = args.pixel_content_weight_warp  # 1.
                if iteration == args.pixel_steps and now_frame == args.start_frame:
                    break
                
            optimizer.zero_grad()
            input = T.clamp(input_img, 0., 1.)
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                content_score, style_score = vgg_loss(input)
                style_score *= style_weight
                content_score *= content_weight
                tv_score = tv_weight * losses.tv_loss(input)

            loss = style_score + content_score + tv_score
            
            
            
            ###############
            # warping score
            ###############
            warped_loss = torch.Tensor([0]).to(device)
            warped_num = 0
            occlusion_mask = None
            new_mask_12 = None
            new_mask_21 = None
            if args.warping_scheme is not None:
                for skip in args.warping_scheme:
                    if now_frame - args.start_frame > skip - 1:
                        now_stroke = input * 255.
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            
                            warped_loss_skip, occlusion_mask, warped_images_next = calc_warping_loss(
                                flow_info[skip]["init_stroke"],
                                now_stroke,
                                flow_info[skip]["flow12"],
                                flow_info[skip]["flow21"],
                                flow_info[skip]["occ_flow12"],
                                flow_info[skip]["occ_flow21"],
                                device,
                                criterion,
                                args.pixel_reduction
                            )
                            
                        
                        warped_loss += warped_loss_skip * warp_weight
                        warped_num += 1
                        
            if warped_num > 0:
                warped_loss = warped_loss / warped_num
                loss += warped_loss[0]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #########
            # logging
            #########
            if iteration % print_freq == 0:
                LOGGER.info(
                    f'[{iteration}/{num_steps}] pixel, style loss:{style_score:.3f}, content loss:{content_score:.3f}, tv loss:{tv_score:.3f}, warped loss:{warped_loss.item():.3f}')
                save_path = os.path.join(
                    root, f'pixel_stylized_forward_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                )
                save = (
                    input[0].detach().cpu().numpy().transpose(
                        1,
                        2,
                        0) *
                    255.).astype(
                    np.uint8)
                save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                save = after_pad(save,pad=args.after_pad)
                cv2.imwrite(save_path, save)
                
        LOGGER.info(f'elapsed time:{datetime.datetime.now()-start_whole}')
    
    ##############################################
    # optimize the canvas pixel-wise warping cycle
    ##############################################
    if args.warping_mode_pixel >=  1:
        for now_frame in range(args.start_frame, args.end_frame):
            style_img_file = args.style_img_file
            content_img_file = f"{args.root_dir}/{str(now_frame).zfill(zfill_length)}.jpg"

            start = datetime.datetime.now()
            LOGGER.info(f'style:{style_img_file}, content:{content_img_file}')

            vgg_weight_file = args.vgg_weight_file


            print_freq = 100
            if args.debug:
                num_steps = 10
            else:
                num_steps = args.pixel_steps_warp_2nd + args.pixel_steps_warp_after

            style_weight = args.pixel_style_weight_warp  # 10000.
            content_weight = args.pixel_content_weight_warp  # 1.
            tv_weight = args.pixel_tv_weight  # 1.#0.008#tv_weight=0
            warp_weight = args.pixel_warp_weight  # 1.#0.008#tv_weight=0
            resize_size = args.pixel_size

            content_img_resized = image_loader(content_img_file, resize_size, device)
            padder = InputPadder(content_img_resized.shape)
            content_img_resized = padder.pad(content_img_resized)[0]
            style_img_resized = F.resize(style_img, resize_size)
            _,_,h,w = content_img_resized.shape
            save_path = os.path.join(
                root, f'pixel_stylized_forward_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
            )
            input_img = image_loader(save_path, resize_size, device)
            input_img = T.nn.Parameter(input_img, requires_grad=True)
            print(input_img.shape,content_img.shape,content_img_resized.shape)

            vgg_loss = losses.StyleTransferLosses(vgg_weight_file, content_img_resized, style_img_resized,
                                                  px_content_layers, px_style_layers)
            vgg_loss.to(device).eval()

            scaler = torch.cuda.amp.GradScaler()


            if args.warping_scheme is not None:
                flow_info = {
                    "forward":{},
                    "backward":{}
                }
                for skip in args.warping_scheme:
                    if now_frame - args.start_frame > skip - 1:
                        init_path = os.path.join(
                            root, f'pixel_stylized_forward_{str(now_frame-skip).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                        flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                            args, resize_size, now_frame, zfill_length, device, init_path, skip=skip
                        )
                        flow_info["forward"][skip] = {
                            "flow12": flow12,
                            "flow21": flow21,
                            "occ_flow12": occ_flow12,
                            "occ_flow21": occ_flow21,
                            "init_stroke": init_stroke,
                        }
                    if args.end_frame - now_frame > skip:
                        init_path = os.path.join(
                            root, f'pixel_stylized_forward_{str(now_frame+skip).zfill(zfill_length)}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                        flow12, flow21, occ_flow12, occ_flow21, init_stroke = flow_setup(
                            args, resize_size, now_frame+skip, zfill_length, device, init_path, skip=skip
                        )
                        flow_info["backward"][skip] = {
                            "flow12": flow12,
                            "flow21": flow21,
                            "occ_flow12": occ_flow12,
                            "occ_flow21": occ_flow21,
                            "init_stroke": init_stroke,
                        }

            if args.occlusion_replace:
                input_img_forward = input_img
                input_img_backward = input_img
                warped_images_forward = None
                warped_images_backward = None
                new_mask_forward = None
                new_mask_backward = None
                for skip in args.warping_scheme:
                    if now_frame - args.start_frame > skip - 1:
                        flow21 = flow_info["forward"][skip]["flow21"]
                        occ_flow21 = flow_info["forward"][skip]["occ_flow21"]
                        save_path = os.path.join(
                            root, f'{args.replace_type}_{os.path.basename(f"{args.root_dir}/{str(now_frame-skip).zfill(zfill_length)}.jpg").split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                        pixel_stroke = image_loader(save_path, resize_size, device) * 255.
                        warped_images_next_21, warped_mask_next = warp(pixel_stroke, flow21, device)
                        new_mask_21 = 1.0 - ((1.0 - warped_mask_next) + occ_flow21).clamp(0, 1)

                        input_img_forward = warped_images_next_21 / 255. * new_mask_21 + \
                            input_img * (1 - new_mask_21)
                        if args.occlusion_replace_main == skip:
                            warped_images_forward = warped_images_next_21
                        if new_mask_forward is None:
                            new_mask_forward = new_mask_21
                        else:
                            new_mask_forward = new_mask_21 * new_mask_forward
                        
                    if args.end_frame - now_frame > skip:
                        flow12 = flow_info["backward"][skip]["flow12"]
                        occ_flow12 = flow_info["backward"][skip]["occ_flow12"]
                        save_path = os.path.join(
                            root, f'{args.replace_type}_{os.path.basename(f"{args.root_dir}/{str(now_frame+skip).zfill(zfill_length)}.jpg").split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                        )
                        pixel_stroke = image_loader(save_path, resize_size, device) * 255.
                        warped_images_next_12, warped_mask_next = warp(pixel_stroke, flow12, device)
                        new_mask_12 = 1.0 - ((1.0 - warped_mask_next) + occ_flow12).clamp(0, 1)

                        input_img_backward = warped_images_next_12 / 255. * new_mask_12 + \
                            input_img * (1 - new_mask_12)
                        if args.occlusion_replace_main == skip:
                            warped_images_backward = warped_images_next_12
                        if new_mask_backward is None:
                            new_mask_backward = new_mask_12
                        else:
                            new_mask_backward = new_mask_12 * new_mask_backward
                        

                if input_img_forward is not None and input_img_backward is not None:
                    input_img = (input_img_forward + input_img_backward) / 2
                elif input_img_forward is None and input_img_backward is not None:
                    input_img = input_img_backward
                elif input_img_forward is not None and input_img_backward is None:
                    input_img = input_img_forward

                input_img = T.nn.Parameter(input_img, requires_grad=True)

            optimizer = optim.Adam([input_img], lr=1e-3)
            LOGGER.info('Optimizing pixel-wise canvas warping..')

            for iteration in range(num_steps):            
                optimizer.zero_grad()
                input = T.clamp(input_img, 0., 1.)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    content_score, style_score = vgg_loss(input)
                    style_score *= style_weight
                    content_score *= content_weight
                    tv_score = tv_weight * losses.tv_loss(input)

                loss = style_score + content_score + tv_score



                ###############
                # warping score
                ###############
                warped_loss = torch.Tensor([0]).to(device)
                warped_num = 0
                occlusion_mask = None
                new_mask_21_for = None
                new_mask_12_for = None
                new_mask_21_back = None
                new_mask_12_back = None

                if args.warping_scheme is not None:
                    for skip in args.warping_scheme:
                        if now_frame - args.start_frame > skip - 1:
                            now_stroke = input * 255.
                            with torch.cuda.amp.autocast(enabled=args.amp):
                                
                                warped_loss_skip, occlusion_mask, warped_images_next = calc_warping_loss(
                                    flow_info["forward"][skip]["init_stroke"],
                                    now_stroke,
                                    flow_info["forward"][skip]["flow12"],
                                    flow_info["forward"][skip]["flow21"],
                                    flow_info["forward"][skip]["occ_flow12"],
                                    flow_info["forward"][skip]["occ_flow21"],
                                    device,
                                    criterion,
                                    args.pixel_reduction
                                )


                            warped_loss += warped_loss_skip * warp_weight
                            warped_num += 1

                        if args.end_frame - now_frame > skip:
                            now_stroke = input * 255.
                            with torch.cuda.amp.autocast(enabled=args.amp):
                                
                                warped_loss_skip, occlusion_mask, warped_images_next = calc_warping_loss(
                                    now_stroke,
                                    flow_info["backward"][skip]["init_stroke"],
                                    flow_info["backward"][skip]["flow12"],
                                    flow_info["backward"][skip]["flow21"],
                                    flow_info["backward"][skip]["occ_flow12"],
                                    flow_info["backward"][skip]["occ_flow21"],
                                    device,
                                    criterion,
                                    args.pixel_reduction
                                )


                            warped_loss += warped_loss_skip * warp_weight
                            warped_num += 1

                if warped_num > 0:
                    warped_loss = warped_loss / warped_num
                    loss += warped_loss[0]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                #########
                # logging
                #########
                if iteration % print_freq == 0:
                    LOGGER.info(
                        f'[{iteration}/{num_steps}] pixel, style loss:{style_score:.3f}, content loss:{content_score:.3f}, tv loss:{tv_score:.3f}, warped loss:{warped_loss.item():.3f}')
                    save_path = os.path.join(
                        root, f'pixel_stylized_warped_{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}.jpg'
                    )
                    save = (
                        input[0].detach().cpu().numpy().transpose(
                            1,
                            2,
                            0) *
                        255.).astype(
                        np.uint8)
                    save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                    save = after_pad(save,pad=args.after_pad)
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
            save = after_pad(save,pad=args.after_pad)
            cv2.imwrite(save_path, save)
            LOGGER.info('Finished!')
            LOGGER.info(f'elapsed time:{datetime.datetime.now()-start}')
    
    LOGGER.info('Finished! ALL !')
    LOGGER.info(f'elapsed time:{datetime.datetime.now()-start_whole}')

    
    video_description = "brushstroke" if args.brushstroke else "normal"
    output2video(
        input_dir=args.root_dir, style_path=style_img_file, names=[
            video_description], roots=[root], save_dir=root, c_name="output_video",
        fps=None,
        zfill_length=zfill_length,
        start=args.start_frame,
        end=args.end_frame,
        save_name=save_name_pixel
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, default="setting")
    args = parser.parse_args()
    params = import_module(args.params)
    main(params.CFG, args.params)

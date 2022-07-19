class CFG:
    ######################
    # initial setting
    ########################
    device = "cuda"
    style_img_file = "samples/styles/starry_night.jpg"
    input_video = "samples/contents/davis_goat.mp4"
    start_frame = 0
    end_frame = 200
    root_dir = None
    output_dir = "results"
    model_name = "pixel_warping_BrushStroke_fromVideos_512p"
    debug = False
    zfill_length = 3
    amp = True
    
    ########################
    # pretrained model
    ########################
    vgg_weight_file = 'pretrained/vgg19_weights_normalized.h5'

    class raft_args:
        model = "pretrained/raft-sintel.pth"
        small = False
        mixed_precision = False
        alternate_corr = False
        dropout = 0

    ########################
    # about training
    ########################
    criterion = "mseloss"
    brushstroke = True

    ########################
    # about warping
    ########################
    warp_mask_use = True
    warping_scheme = [1, 8, 16, 32]
    warping_mode_stroke = 0  # 0 : forward, 1: forward+backward+cycle
    warping_mode_pixel = 0  # 0 : forward, 1: forward+backward+cycle

    ########################
    # stroke optimize setting
    ########################
    stroke_img_size = 256
    canvas_color = "gray"
    num_strokes = 5000
    samples_per_curve = 10
    brushes_per_pixel = 20
    stroke_steps = 50
    stroke_steps_warp = 0
    brush_initialize = True

    stroke_reduction = "sum"
    
    stroke_style_weight = 3
    stroke_style_weight_warp = 3e-1
    stroke_content_weight = 1
    stroke_content_weight_warp = 1e0
    stroke_tv_weight = 0.008
    stroke_curv_weight = 4
    stroke_warp_weight = 0  

    
    ########################
    # ixel optimize setting
    ########################
    training_stop = 1.5
    training_stop_init = 15
    pixel_size = 512
    pixel_steps = 2700
    pixel_steps_warp = 301
    pixel_steps_warp_1st = 301
    pixel_steps_warp_2nd = 501
    pixel_steps_warp_after = 0
    
    pixel_reduction = "mean"
    
    pixel_style_weight = 1e3  
    pixel_style_weight_warp = 1e3
    pixel_content_weight = 1e0
    pixel_content_weight_warp = 1e0
    pixel_tv_weight = 4e3
    pixel_warp_weight = 1e5
    
    after_pad = 2
    
    
    

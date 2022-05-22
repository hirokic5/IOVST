class CFG:
    #################
    # initial setting
    #################
    device = "cuda"
    style_img_file = "data/images/girl-on-a-divan.jpg"
    input_video = None  # "data/videos/traffic.mp4"
    start_frame = 0
    end_frame = 50
    root_dir = "data/traffic_video"
    output_dir = "results/traffic"
    model_name = "pixel_warping_BrushStroke_fromVideos_512p"
    debug = False
    zfill_length = 3

    ##################
    # pretrained model
    ##################
    vgg_weight_file = 'pretrained/vgg19_weights_normalized.h5'
    class raft_args:
        model = "pretrained/raft-things.pth"
        small = False
        mixed_precision = False
        alternate_corr = False
        dropout = 0

    ################
    # about training
    ################
    criterion = "mseloss"
    brushstroke = True
    
    ###############
    # about warping
    ###############
    warp_mask_use = True
    warping_scheme = [1,4,16]

    #########################
    # stroke optimize setting
    #########################
    initial_input = None 
    stroke_img_size = 256
    canvas_color = "gray"
    num_strokes = 5000
    samples_per_curve = 10
    brushes_per_pixel = 20
    stroke_steps = 100
    stroke_steps_warp = 100
    brush_initialize = True

    stroke_style_weight = 3
    stroke_content_weight = 1
    stroke_style_weight_warp = 3e-1
    stroke_content_weight_warp = 1e-1
    stroke_tv_weight = 0.008
    stroke_curv_weight = 4
    stroke_warp_weight = 1e3
    stroke_skip_warp_weight = 1e3

    fusion_stroke = True
    weight_save = True
    ########################
    # pixel optimize setting
    ########################
    initial_input = None
    pixel_style_weight = 1e4
    pixel_content_weight = 1e0
    pixel_tv_weight = 1e2
    pixel_warp_weight = 1e4
    pixel_skip_warp_weight = 1e4
    pixel_size = 512
    pixel_steps = 1500
    pixel_steps_warp = 1000
    pixel_style_weight_warp = 1e2
    pixel_content_weight_warp = 1e-1

    
    
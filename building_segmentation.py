from time import time
from model import EfficientFPN
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from dataset import test_provider
from os.path import join
import os
from utils import time_dif

def segment_buildings(root_dir, cfg, **kwargs):
    start = time()
    torch.cuda.empty_cache()
    model = EfficientFPN(encoder_name=cfg.seg_model_name,
                        use_context_block=cfg.use_context_block,
                        use_mish=cfg.use_mish,
                        use_attention=cfg.use_attention)   
                        
    model_dir = join(root_dir,cfg.model_dir)
    state = torch.load(join(model_dir,cfg.seg_model_weights),
                    map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Device used ' + torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        print('Device used CPU')

    model = model.to(device)
    model = model.eval()

    os.mkdir('submission_format')
    os.mkdir('segmentation')
    
    if cfg.show_intermediate:
        os.mkdir('bce')
        os.mkdir('dice')
        os.mkdir('dice_over_thresh')

    DATA_DIR = os.path.join(root_dir,cfg.data_dir)
    print(DATA_DIR)
    dataloader = test_provider(
        DATA_DIR,
        batch_size = 1,
        num_workers = 0,
        train_width = cfg.tile_size_x,
        train_height = cfg.tile_size_y,
        phase = 'test',
        scale_factor = cfg.scale_factor)

    if cfg.mine_empty and not os.path.exists(cfg.output_dir_empty):
        os.mkdir(cfg.output_dir_empty)
    
    tk0 = tqdm(dataloader)
    with torch.no_grad():
        for itr, batch in enumerate(tk0):    # replace `dataloader` with `tk0` for tqdm
            images, masks, idx, sf  = batch
            base_name = os.path.basename(idx[0])
            x = cv2.imread(idx[0]).shape[0]
            y = cv2.imread(idx[0]).shape[1]

            if cfg.flip > 10:
                images = images.flip([2, 3])
            elif cfg.flip != 0:
                images = images.flip(cfg.flip)
            if cfg.transpose:
                images = images.permute(0, 1, 3, 2)

            images = images.to(device)
                
            outputs, dice, cls, scale = model(images, masks)
            cls = F.softmax(cls, 1)
    
            if cfg.flip > 10:
                outputs = outputs.flip([2, 3])
                dice = dice.flip([2, 3])
            elif cfg.flip != 0:
                outputs = outputs.flip(cfg.flip)
                dice = dice.flip(cfg.flip)
            if cfg.transpose:
                outputs = outputs.permute(0, 1, 3, 2)
                dice = dice.permute(0, 1, 3, 2)

            outputs_log = F.softmax(outputs, 1)
            max_cls = outputs_log[:, 1].max().item()  
            if cfg.mine_empty and cls[0, 1] < cfg.empty_threshold and max_cls < cfg.empty_threshold:
                dimg = cv2.imread(idx[0])
                print(f'empty image {idx[0]}')
                if not os.path.exists(f'{cfg.output_dir_empty}/{os.path.basename(base_name)}'):
                    cv2.imwrite(f'{cfg.output_dir_empty}/{base_name}', dimg)


            if cfg.show_intermediate:
                outputs_log2 = F.interpolate(outputs_log, size=(x, y), mode='bilinear', align_corners=False)
                # outputs_log2 = F.interpolate(outputs_log, size=(1024, 1024), mode='bilinear', align_corners=False)
                # outputs_log_out = outputs_log2[:, 1].cpu().numpy() * 255
                # outputs_log_out = outputs_log_out[0].astype(np.uint8)
                outputs_log_out = outputs_log2[0, 0].cpu().numpy() * 255
                outputs_log_out = outputs_log_out.astype(np.uint8)
                cv2.imwrite(f'bce/{base_name}', outputs_log_out)
        
                dice = torch.sigmoid(dice)
                dice = F.interpolate(dice, size=(x, y), mode='bilinear', align_corners=False)
                dice_out = dice[0, 0].cpu().numpy() * 255
                dice_out = dice_out.astype(np.uint8)
                cv2.imwrite(f'dice/{base_name}', dice_out)
                
                dice_at = dice[dice < 0.5] = 0
                dice_at = dice[dice >= 0.5] = 1
                dice_at_out = dice_at[0, 0].cpu().numpy() * 255
                # dice = dice[0, 0, :, :].cpu().numpy() * 255
                dice_at_out = dice_at_out.astype(np.uint8)
                cv2.imwrite(f'dice_over_thresh/{base_name}', dice_at_out)
    
            outputs_log = F.interpolate(outputs_log, size=(x, y), mode='bilinear', align_corners=False)
            outputs_log = outputs_log[:, 1]
            avg = outputs_log
            avg[avg < cfg.threshold] = 0
            avg[avg >= cfg.threshold] = 1 
            avg = avg.cpu().numpy() * 255
            avg = avg.astype(np.uint8)
            avg = avg[0, :, :]      
            cv2.imwrite(f'submission_format/{base_name}', avg)

    print("Building segmentation done in {}".format(time_dif(start)))

import torch
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml # For loading YAML config
from yacs.config import CfgNode # For creating config object

# Adjust imports based on the new project structure
# Assuming train.py is in scripts/, dataset.py is in datasets/, etc.
import sys
# Add project root to path to allow imports like datasets.dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.insert(0, project_root)

from datasets.dataset import forgeryHSIDataset, normColor # Import the dataset class and visualization helper
from models.hrnet import HRNetV2 # Import the modified model
from utils.metrics import SegMetrics # Import the metrics class
# utils.transforms might not be needed if transforms are in __getitem__
# from utils.utils import Denormalize # Denormalize might not be needed if using normColor

def checkPerformance(model, device, cfg):
    """
    Evaluates the model performance on the validation set.
    Mirrors the logic from DFCN_Py_all's performance_eval.py checkPerformance function.
    
    Args:
        model: The model to evaluate (already on the correct device).
        device: The device (e.g., 'cuda:0') to run evaluation on.
        cfg: The configuration object (loaded from YAML).
    
    Returns:
        A dictionary containing evaluation metrics (from SegMetrics).
    """
    print("\n--- Starting Validation Run ---")
    
    # Prepare validation dataset
    # Construct absolute path for list file based on script location
    val_list_path = os.path.join(project_root, cfg.DATASET.Val_data.replace("../", ""))
    print(f"[INFO] Loading validation data list from: {val_list_path}")
    
    # Note: DFCN's validation transform is basic. No random augmentations.
    # The dataset class itself handles the crop and normalization needed.
    val_dataset = forgeryHSIDataset(
        root=cfg.DATASET.ROOT, 
        flist=val_list_path,  # 直接传入完整路径
        split='val'
    )
    if len(val_dataset) == 0:
        print("[ERROR] Validation dataset is empty. Check Val_data path and file content.")
        return { # Return default/error score
            "Overall acc": 0.0,
            "Mean acc": 0.0,
            "Mean IoU": 0.0,
            "Class IoU": {0: 0.0, 1: 0.0}
        }
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.TEST.BATCH_SIZE, 
        shuffle=False, # No shuffling for validation
        num_workers=cfg.TEST.NUM_WORKERS, 
        pin_memory=True, # Use pin_memory if possible
        drop_last=False # Don't drop last batch in validation
    )
    
    # Initialize metrics computer
    metric = SegMetrics(cfg.DATASET.NUM_CLASSES, device)
    
    # Results path for saving images
    save_img_dir = os.path.join(project_root, cfg.TEST.RESULTS_PATH.replace("./", ""))
    save_img_count = cfg.TEST.RESULTS_NUM
    if save_img_count > 0:
        os.makedirs(save_img_dir, exist_ok=True) # Create dir if it doesn't exist
        print(f"[INFO] Validation images will be saved to: {save_img_dir}")
    img_save_idx = 0 # Counter for saved images
        
    model.eval() # Set model to evaluation mode
    
    # Create buffer for saving visualization image (Image | Label | Prediction)
    # 使用實際圖像尺寸(256x256)而不是裁剪尺寸(224x224)
    vis_height = 256
    vis_width = 256 * 3
    img_buffer = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    pbar = tqdm(val_loader, desc='Validating')
    for i, data in enumerate(pbar):
        with torch.no_grad(): # Disable gradient calculation
            # 解包数据（与train.py保持一致）
            if len(data) == 4:  # 如果有四个返回值 (image, label, band_label, img_name)
                image, target, bands, img_names = data
            elif len(data) == 3:  # 兼容三个返回值 (image, label, band_label)
                image, target, bands = data
                img_names = None
            else:  # 处理其他可能的情况
                raise ValueError(f"Unexpected dataset return format: {len(data)} items")
            
            image = image.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long) # Target mask
            # bands label is not used in evaluation metric calculation
            
            # Model forward pass (expects dual output now)
            try:
                output_seg, output_bands = model(image)
            except Exception as e:
                print(f"[ERROR] Model forward pass failed during validation: {e}")
                continue # Skip this batch
                
            # Get segmentation prediction by taking argmax
            # output_seg shape: (B, NumClasses, H, W)
            pred = output_seg.detach().max(dim=1)[1] # Get class index with max score -> (B, H, W)
            
            # Update confusion matrix
            metric.update(target, pred)
            
            # Save visualization images periodically
            if save_img_count > 0 and img_save_idx < save_img_count and i * cfg.TEST.BATCH_SIZE < len(val_loader.dataset):
                 # Check if there are images left in the batch to process
                if image.size(0) > 0: 
                    try:
                        # Take the first image from the batch
                        img_to_save = image[0].detach().cpu().numpy() # (C, H, W)
                        tgt_to_save = target[0].cpu().numpy()       # (H, W)
                        prd_to_save = pred[0].cpu().numpy()         # (H, W)
                        
                        # Transpose image: (C, H, W) -> (H, W, C)
                        img_to_save = img_to_save.transpose(1, 2, 0)
                        
                        # Renormalize image from [-1, 1] to [0, 1] then scale to [0, 255]
                        img_display = ((img_to_save + 1.0) / 2.0) 
                        img_display = np.clip(img_display, 0, 1) # Clip just in case
                        # Use normColor for visualization (selects 3 bands)
                        img_display_rgb = normColor(img_display * 255) # normColor expects 0-255 range ideally
                        
                        # Decode target and prediction masks to color images
                        try:
                            tgt_color = val_loader.dataset.decode_target(tgt_to_save).astype(np.uint8)
                            prd_color = val_loader.dataset.decode_target(prd_to_save).astype(np.uint8)
                        except AttributeError as e:
                            print(f"[WARN] 缺少 decode_target 方法: {e}")
                            # 簡單的後備方案 - 為缺少時創建一個簡單的灰度圖
                            tgt_color = np.stack([tgt_to_save*255, tgt_to_save*255, tgt_to_save*255], axis=2).astype(np.uint8)
                            prd_color = np.stack([prd_to_save*255, prd_to_save*255, prd_to_save*255], axis=2).astype(np.uint8)
                        
                        # Fill the image buffer
                        img_buffer[:, :256, :] = img_display_rgb
                        img_buffer[:, 256:512, :] = tgt_color
                        img_buffer[:, 512:, :] = prd_color
                        
                        # Save the combined image
                        save_path = os.path.join(save_img_dir, f"val_res_{img_save_idx}.png")
                        Image.fromarray(img_buffer).save(save_path)
                        img_save_idx += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to process or save validation image {img_save_idx}: {e}")

    # Get final scores from metrics computer
    score = metric.get_results()
    
    print(f"--- Validation Results ---")
    print(f"  Overall Accuracy: {score['Overall acc']:.4f}")
    print(f"  Mean Accuracy: {score['Mean acc']:.4f}")
    print(f"  Mean IoU: {score['Mean IoU']:.4f}")
    # Safely print class IoU
    cls_iou_str = {k: f'{v:.4f}' for k, v in score['Class IoU'].items()} 
    print(f"  Class IoU: {cls_iou_str}")
    print(f"-------------------------")
    
    # model.train() # Set model back to training mode (should be done in the main training loop)
    return score

# Helper function to load config from YAML
def load_config_yaml(config_path):
    # 處理相對路徑
    if not os.path.isabs(config_path):
        # 如果是相對路徑，則相對於項目根目錄
        abs_config_path = os.path.join(project_root, config_path)
    else:
        abs_config_path = config_path
    
    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"Config file not found: {abs_config_path}")
    print(f"[INFO] Loading config from: {abs_config_path}")
    
    # 使用臨時YAML配置文件
    temp_yaml_path = os.path.join(project_root, "temp", "temp_config.yaml")
    os.makedirs(os.path.dirname(temp_yaml_path), exist_ok=True)
    
    # 複製原始YAML文件
    import shutil
    shutil.copy2(abs_config_path, temp_yaml_path)
    
    # 獲取基礎配置
    from config.default import get_cfg
    cfg = get_cfg()
    
    # 從文件加載
    cfg.merge_from_file(temp_yaml_path)
    
    return cfg

# Example main function for standalone testing (optional)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Evaluation Script (DFCN Style)")
    # Adjust default path relative to this script's location
    parser.add_argument("--config_path", type=str, default="../config/train_config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    args = parser.parse_args()

    # Load configuration
    try:
        cfg = load_config_yaml(args.config_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load or parse config file {args.config_path}: {e}")
        sys.exit(1)
        
    # Setup device
    if cfg.CUDA.USE_CUDA and torch.cuda.is_available():
        if not cfg.CUDA.CUDA_NUM:
             print("[WARN] CUDA specified but no GPU IDs listed in CUDA.CUDA_NUM. Using GPU 0.")
             device_id = 0
        else:
             device_id = cfg.CUDA.CUDA_NUM[0] # Use the first GPU for evaluation
        device = torch.device(f"cuda:{device_id}")
        print(f"[INFO] Using GPU {device_id} for evaluation.")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU for evaluation.")

    # Load Model
    print("[INFO] Initializing model...")
    try:
        model = HRNetV2(
            C=cfg.MODEL.C,
            num_class=cfg.DATASET.NUM_CLASSES,
            in_ch=cfg.MODEL.IN_CHANNELS,
            use3D=(cfg.MODEL.USE_3D == True),
            useAttention=(cfg.MODEL.USE_ATTENTION == True)
        ).to(device)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        sys.exit(1)
        
    # Load checkpoint weights
    checkpoint_path = os.path.join(project_root, args.checkpoint.replace("./", ""))
    if os.path.isfile(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle potential full checkpoint vs state_dict only
            state_dict = checkpoint.get('model_state_dict', checkpoint) 
            
            # Handle DataParallel wrapper prefix ('module.') if present
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict)
            print("[INFO] Checkpoint loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint from {checkpoint_path}: {e}")
            sys.exit(1)
    else:
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # Run evaluation
    results = checkPerformance(model, device, cfg)
    print("\nFinal Standalone Evaluation Results:")
    # Ensure results dict is not None before printing
    if results:
        print(results)
    else:
        print("Evaluation failed.") 
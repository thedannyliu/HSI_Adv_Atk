import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm, trange # Use trange for epoch loop
import time
import sys
from collections import OrderedDict
import random

# Multi-GPU and SyncBatchNorm Support (Ensure sync_batchnorm is installed)
try:
    from sync_batchnorm import convert_model, DataParallelWithCallback
    sync_bn_avail = True
except ImportError:
    print("[WARN] sync_batchnorm not found. Multi-GPU SyncBatchNorm disabled. Falling back to nn.DataParallel.")
    sync_bn_avail = False
    # Use standard DataParallel as fallback
    from torch.nn import DataParallel as DataParallelWithCallback 
    # convert_model equivalent (does nothing if not using sync_bn)
    def convert_model(model):
        return model 

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Local imports (relative to scripts/)
from datasets.dataset import forgeryHSIDataset, normColor # Using the new dataset loader
from models.hrnet import HRNetV2 # Using the modified HRNetV2 model
from config.default import CfgNode # 從我們的default.py導入CfgNode
from utils.metrics import SegMetrics # Using the DFCN metrics class
from scripts.performance_eval import checkPerformance, load_config_yaml # Import validation function and YAML loader

# --- Loss Functions (Copied/Adapted from DFCN train.py) ---
class DiceBCELoss(nn.Module):
    # As defined in DFCN train.py
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float() # Ensure target is float for calculations
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

class IoULoss(nn.Module):
    # As defined in DFCN train.py
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU

# Note: BoundaryLoss is complex and wasn't used in DFCN's main loss calculation, 
# so it's omitted here for simplicity unless specifically requested.

# --- Main Training Function --- 
def main(args):
    # Load Configuration from YAML
    try:
        print(f"[INFO] Attempting to load config from: {args.config_path}")
        cfg = load_config_yaml(args.config_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load or parse config file {args.config_path}: {str(e)}")
        # Print the full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Setup Device(s)
    use_cuda = cfg.CUDA.USE_CUDA and torch.cuda.is_available()
    gpu_ids = cfg.CUDA.CUDA_NUM
    if use_cuda:
        if not gpu_ids:
            print("[WARN] USE_CUDA is True, but CUDA.CUDA_NUM is empty. Using default GPU 0.")
            gpu_ids = [0]
        base_device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(base_device) # Set default CUDA device
        print(f"[INFO] Using CUDA. Primary GPU: {gpu_ids[0]}. Available GPUs: {torch.cuda.device_count()}")
        print(f"[INFO] Training will use GPU IDs: {gpu_ids}")
    else:
        base_device = torch.device("cpu")
        print("[INFO] Using CPU")

    # Set Random Seed
    seed = cfg.GENERAL.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) # Seed python's random module used in dataset
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        # Consider adding these for full reproducibility, but they can slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Prepare Dataset & DataLoader
    train_list_path = os.path.join(project_root, cfg.DATASET.Train_data.replace("../", ""))
    print(f"[INFO] Loading training data list from: {train_list_path}")
    # Transforms are mostly handled inside dataset's __getitem__ based on DFCN logic
    train_dataset = forgeryHSIDataset(
        root=cfg.DATASET.ROOT, 
        flist=train_list_path,  # 直接传入完整路径
        split='train'
    )
    if len(train_dataset) == 0:
         print("[ERROR] Training dataset is empty. Aborting.")
         sys.exit(1)
         
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE * len(gpu_ids) if use_cuda else cfg.TRAIN.BATCH_SIZE, # Total batch size across GPUs
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY, 
        drop_last=True
    )
    print(f"[INFO] Training dataset size: {len(train_dataset)}, Loader batch size: {cfg.TRAIN.BATCH_SIZE * len(gpu_ids) if use_cuda else cfg.TRAIN.BATCH_SIZE}")

    # Model Setup
    print("[INFO] Setting up model...")
    model = HRNetV2(
        C=cfg.MODEL.C,
        num_class=cfg.DATASET.NUM_CLASSES,
        in_ch=cfg.MODEL.IN_CHANNELS,
        use3D=(cfg.MODEL.USE_3D == True),
        useAttention=(cfg.MODEL.USE_ATTENTION == True)
    )

    # --- Multi-GPU Handling --- 
    if use_cuda and len(gpu_ids) > 1:
        print(f"[INFO] Using {len(gpu_ids)} GPUs. Applying Data Parallelism...")
        if sync_bn_avail:
             print("[INFO]   using SyncBatchNorm.")
             model = convert_model(model) # Convert BN layers for SyncBN
             model = model.to(base_device) # Move model to primary GPU *before* wrapping
             model = DataParallelWithCallback(model, device_ids=gpu_ids) # Use SyncBN wrapper
        else:
             print("[INFO]   using standard nn.DataParallel (SyncBatchNorm not available).")
             model = model.to(base_device)
             model = nn.DataParallel(model, device_ids=gpu_ids) # Use standard DataParallel
    else:
        model = model.to(base_device) # Single GPU or CPU

    # Optimizer (Using AdamW as per DFCN's code)
    print("[INFO] Setting up Optimizer...")
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.TRAIN.LEARNING_RATE, 
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    print(f"[INFO] Optimizer: AdamW, LR: {cfg.TRAIN.LEARNING_RATE}, Weight Decay: {cfg.TRAIN.WEIGHT_DECAY}")

    # Scheduler (Using MultiStepLR as per DFCN's code)
    print("[INFO] Setting up Learning Rate Scheduler...")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=cfg.TRAIN.SCHEDULER.MILESTONES, 
        gamma=cfg.TRAIN.SCHEDULER.GAMMA
    )
    print(f"[INFO] Scheduler: MultiStepLR, Milestones: {cfg.TRAIN.SCHEDULER.MILESTONES}, Gamma: {cfg.TRAIN.SCHEDULER.GAMMA}")

    # Loss Function
    print("[INFO] Setting up Loss Function...")
    criterion = None
    if cfg.TRAIN.LOSS.NAME == "CrossEntropyLoss":
        class_weights = torch.tensor(cfg.TRAIN.LOSS.WEIGHTS).to(base_device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[INFO] Using CrossEntropyLoss with weights: {cfg.TRAIN.LOSS.WEIGHTS}")
    elif cfg.TRAIN.LOSS.NAME == "DiceBCELoss":
        criterion = DiceBCELoss()
        print("[INFO] Using DiceBCELoss")
    elif cfg.TRAIN.LOSS.NAME == "IoULoss":
        criterion = IoULoss()
        print("[INFO] Using IoULoss")
    else:
        raise ValueError(f"Unsupported loss function: {cfg.TRAIN.LOSS.NAME}")

    # Auxiliary Loss for Band Prediction (if enabled)
    criterion_aux = None
    if cfg.TRAIN.BAND_REG:
        criterion_aux = nn.BCEWithLogitsLoss() # As used in DFCN
        print("[INFO] Auxiliary Band Regularization Loss Enabled (BCEWithLogitsLoss)")

    # Load Checkpoint (If specified)
    start_epoch = cfg.TRAIN.EPOCH_START
    checkpoint_path = os.path.join(project_root, cfg.TRAIN.CHECKPOINT.replace("./", "")) 
    if cfg.TRAIN.CHECKPOINT and os.path.isfile(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=base_device)
            # Handle DataParallel prefix if needed
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
            if 'optimizer_state_dict' in checkpoint:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 print("[INFO]   Optimizer state loaded.")
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 print("[INFO]   Scheduler state loaded.")
            if 'epoch' in checkpoint:
                 start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
                 print(f"[INFO]   Resuming from epoch {start_epoch}")
            else:
                print("[WARN] Checkpoint loaded but no epoch found, starting from configured start epoch.")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint correctly: {e}. Starting from scratch.")
            start_epoch = cfg.TRAIN.EPOCH_START
    else:
        print("[INFO] No valid checkpoint specified, starting training from scratch.")

    # Prepare Logging (TensorBoard)
    log_dir = os.path.join(project_root, cfg.TRAIN.LOG_PATH.replace("./", ""))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

    # Prepare Checkpoint Saving Directory
    save_dir = os.path.join(project_root, cfg.TRAIN.SAVE_WEIGHT_PATH.replace("./", ""))
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Model checkpoints will be saved to: {save_dir}")

    # Training Loop
    print("="*30)
    print("      Starting Training (DFCN Style)      ")
    print("="*30)
    
    max_iou = 0.0 # Track best validation mIoU
    
    epoch_iterator = trange(start_epoch, cfg.TRAIN.EPOCH_END + 1, desc="Epochs")
    for epoch in epoch_iterator:
        model.train() # Set model to training mode
        avg_loss_list = []
        avg_aux_loss_list = [] # Only used if BAND_REG is True
        batch_times = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} Loss=N/A', leave=False)
        for i, data in enumerate(pbar):
            start_time = time.time()
            step = epoch * len(train_loader) + i # Global step for logging
            
            # 解包数据集返回的值
            if len(data) == 4:  # 如果有四个返回值 (image, label, band_label, img_name)
                image, label, label2, img_names = data
            elif len(data) == 3:  # 兼容三个返回值 (image, label, band_label)
                image, label, label2 = data
                img_names = None
            else:  # 处理其他可能的情况
                raise ValueError(f"Unexpected dataset return format: {len(data)} items")
            
            image = image.to(base_device, dtype=torch.float32)
            label = label.to(base_device, dtype=torch.long) # Mask for main loss
            label2 = label2.to(base_device, dtype=torch.float32) # Band labels for aux loss
            
            optimizer.zero_grad()
            
            # Forward pass (Model returns two outputs)
            pred_seg, pred_bands = model(image)
            
            # --- Calculate Main Segmentation Loss --- 
            loss_main = 0
            # For CrossEntropy: Input (B, C, H, W), Target (B, H, W) Long
            if isinstance(criterion, nn.CrossEntropyLoss):
                 loss_main = criterion(pred_seg, label)
            # For Dice/IoU: Input (B, 1, H, W) or (B, H, W) logits, Target (B, H, W) Float
            else:
                 # Assume pred_seg is (B, 2, H, W), need to get class 1 logits/probs
                 # If using BCE-based losses (DiceBCE, IoU), sigmoid is applied inside loss
                 # So we need the raw logit for class 1. 
                 # pred_seg[:, 1] gives logits for class 1. Shape (B, H, W)
                 if pred_seg.size(1) == 2:
                     loss_main = criterion(pred_seg[:, 1], label.float())
                 else: # Should not happen if model output is correct
                      print("[WARN] Loss calculation mismatch: Expecting 2 channels from model for Dice/IoU.")
                      loss_main = torch.tensor(0.0).to(base_device) 
                      
            # --- Calculate Auxiliary Band Loss (if enabled) --- 
            loss_aux = torch.tensor(0.0).to(base_device)
            if cfg.TRAIN.BAND_REG and criterion_aux is not None:
                 loss_aux = criterion_aux(pred_bands, label2)
                 total_loss = loss_main + loss_aux
            else:
                 total_loss = loss_main
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            avg_loss_list.append(loss_main.item())
            if cfg.TRAIN.BAND_REG:
                 avg_aux_loss_list.append(loss_aux.item())
            
            # Update progress bar description
            desc_str = f"Epoch {epoch} Loss={np.mean(avg_loss_list):.4f}"
            if cfg.TRAIN.BAND_REG:
                desc_str += f" (Aux={np.mean(avg_aux_loss_list):.4f})"
            pbar.set_description(desc_str)
            pbar.refresh()
            
            # Log loss to TensorBoard periodically
            if (i + 1) % cfg.TRAIN.LOG_LOSS == 0:
                writer.add_scalar('Loss/train_main', loss_main.item(), step)
                if cfg.TRAIN.BAND_REG:
                     writer.add_scalar('Loss/train_aux', loss_aux.item(), step)
                writer.add_scalar('Loss/train_total', total_loss.item(), step)
            
            # Log images to TensorBoard periodically
            if (i + 1) % cfg.TRAIN.LOG_IMAGE == 0:
                try:
                    img_vis = image[0].detach().cpu().numpy() # C, H, W
                    lbl_vis = label[0].detach().cpu().numpy() # H, W
                    prd_vis = pred_seg[0].detach().max(dim=0)[1].cpu().numpy() # H, W (argmax prediction)
                    
                    # 規範化影像數據範圍為0-255
                    img_vis_rgb = normColor( ((img_vis.transpose(1, 2, 0) + 1.0) / 2.0) * 255.0 ) # H, W, C
                    
                    # 確保 decode_target 方法存在並正確運行
                    try:
                        lbl_vis_rgb = train_loader.dataset.decode_target(lbl_vis).astype(np.uint8) # H, W, C
                        prd_vis_rgb = train_loader.dataset.decode_target(prd_vis).astype(np.uint8) # H, W, C
                    except AttributeError as e:
                        print(f"[WARN] 缺少 decode_target 方法: {e}")
                        # 簡單的後備方案 - 為缺少時創建一個簡單的灰度圖
                        lbl_vis_rgb = np.stack([lbl_vis*255, lbl_vis*255, lbl_vis*255], axis=2).astype(np.uint8)
                        prd_vis_rgb = np.stack([prd_vis*255, prd_vis*255, prd_vis*255], axis=2).astype(np.uint8)
                    
                    # 確保所有圖像都是正確的格式（HWC, uint8）
                    if not (img_vis_rgb.ndim == 3 and img_vis_rgb.shape[2] == 3):
                        print(f"[WARN] 輸入影像格式不正確 (shape={img_vis_rgb.shape})，無法記錄到 TensorBoard")
                    else:
                        writer.add_image('Images/Train_Input', img_vis_rgb, step, dataformats='HWC')
                        writer.add_image('Images/Train_Label', lbl_vis_rgb, step, dataformats='HWC')
                        writer.add_image('Images/Train_Pred', prd_vis_rgb, step, dataformats='HWC')
                except Exception as e:
                    print(f"[WARN] 無法記錄影像到 TensorBoard: {str(e)}")
                    import traceback
                    traceback.print_exc() # 打印詳細的堆棧跟踪，幫助調試

            batch_times.append(time.time() - start_time)
        # End of Epoch Train Loop
        
        # Log average epoch losses
        avg_epoch_loss = np.mean(avg_loss_list)
        writer.add_scalar('Loss/train_epoch_main_avg', avg_epoch_loss, epoch)
        if cfg.TRAIN.BAND_REG:
             avg_epoch_aux_loss = np.mean(avg_aux_loss_list)
             writer.add_scalar('Loss/train_epoch_aux_avg', avg_epoch_aux_loss, epoch)
             print(f"Epoch {epoch} Training Avg Loss: Main={avg_epoch_loss:.4f}, Aux={avg_epoch_aux_loss:.4f}")
        else:
             print(f"Epoch {epoch} Training Avg Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch} Average Batch Time: {np.mean(batch_times):.4f}s")
             
        # --- Validation and Saving --- 
        if (epoch) % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.EPOCH_END:
             val_score = checkPerformance(model, base_device, cfg)
             current_iou = val_score.get("Mean IoU", 0.0)
             current_acc = val_score.get("Overall acc", 0.0)
             current_macc = val_score.get("Mean acc", 0.0)
             
             # Log validation metrics to TensorBoard
             writer.add_scalar('Metrics/val_mean_iou', current_iou, epoch)
             writer.add_scalar('Metrics/val_overall_acc', current_acc, epoch)
             writer.add_scalar('Metrics/val_mean_acc', current_macc, epoch)
             
             # Save last model
             save_path_last = os.path.join(save_dir, 'last.pth')
             print(f"[INFO] Saving last checkpoint to {save_path_last}")
             torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'val_mean_iou': current_iou, 
             }, save_path_last)
             
             # Save best model based on Mean IoU
             if current_iou > max_iou:
                 max_iou = current_iou
                 save_path_best = os.path.join(save_dir, 'best.pth')
                 print(f"[INFO] New best Mean IoU: {max_iou:.4f}. Saving best checkpoint to {save_path_best}")
                 torch.save({
                     'epoch': epoch,
                     'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'val_mean_iou': max_iou, 
                 }, save_path_best)
                 
             print(f"[INFO] Epoch {epoch} Validation: Mean IoU={current_iou:.4f}, Overall Acc={current_acc:.4f}. Best Mean IoU={max_iou:.4f}")
        
        # Step the scheduler
        scheduler.step()
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
    # End of Training Loop
    writer.close() # Close TensorBoard writer
    print("="*30)
    print(f"      Training Finished. Best Validation Mean IoU: {max_iou:.4f}      ")
    print("="*30)

# --- Argument Parser and Entry Point --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperForensics++ Training Script (DFCN Style)")
    # Use config path relative to project root for consistency
    parser.add_argument("--config_path", type=str, default="./config/train_config.yaml",
                        help="Path to the YAML configuration file relative to project root.")
    # Remove --configs argument as data lists are now defined in the main config file
    # parser.add_argument("--configs", type=str, nargs="+", help="Not used in DFCN style. Define data lists in config.", default=None)
    
    args = parser.parse_args()
    main(args) 
from yacs.config import CfgNode as CN

_C = CN()

# 基本設置
_C.GENERAL = CN()
_C.GENERAL.SEED = 42              # 隨機種子
_C.GENERAL.LOG_DIR = "./logs"     # 日誌目錄

# CUDA相關設置
_C.CUDA = CN()
_C.CUDA.USE_CUDA = True           # 是否使用CUDA
_C.CUDA.CUDA_NUM = [0]            # 使用的GPU編號

# 數據集設置
_C.DATASET = CN()
_C.DATASET.ROOT = "/ssd5/HyperForensics/FastHyIn"  # 數據根目錄
_C.DATASET.SPLIT_ROOT = "/ssd5/HyperForensics/FastHyIn/dataset_split_txt"  # 數據集分割檔案根目錄
_C.DATASET.Train_data = "train_cfg0.txt"           # 訓練數據文件列表
_C.DATASET.Val_data = "test_cfg0.txt"              # 驗證數據文件列表
_C.DATASET.NUM_CLASSES = 2                         # 類別數量
_C.DATASET.INFO_CSV = ""                           # 信息CSV文件路徑
_C.DATASET.CONFIGS = [                            # 數據集配置列表
    CN({
        "name": "Origin",
        "train": "Origin/train.txt",
        "val": "Origin/val.txt",
        "test": "Origin/test.txt"
    }),
    CN({
        "name": "config1",
        "train": "Config1/train.txt",
        "val": "Config1/val.txt",
        "test": "Config1/test.txt"
    })
]

# 模型設置
_C.MODEL = CN()
_C.MODEL.NAME = "HRNetV2"          # 模型名稱
_C.MODEL.C = 12                    # HRNetV2的通道數參數
_C.MODEL.IN_CHANNELS = 172         # 輸入通道數
_C.MODEL.INPUT_CHANNELS = 172      # 另一個輸入通道數配置 (與hrnet.py中的參數對應)
_C.MODEL.OUTPUT_CHANNELS = 1       # 輸出通道數
_C.MODEL.USE_3D = True             # 是否使用3D卷積
_C.MODEL.USE_ATTENTION = False     # 是否使用注意力機制
_C.MODEL.PRETRAINED = False        # 是否使用預訓練模型

# 訓練設置
_C.TRAIN = CN()
_C.TRAIN.EPOCH_START = 0           # 開始訓練的輪次
_C.TRAIN.EPOCH_END = 100           # 結束訓練的輪次
_C.TRAIN.BATCH_SIZE = 2            # 批大小
_C.TRAIN.LEARNING_RATE = 0.0005    # 學習率
_C.TRAIN.WEIGHT_DECAY = 0.0001     # 權重衰減
_C.TRAIN.NUM_WORKERS = 4           # 數據加載進程數
_C.TRAIN.PIN_MEMORY = True         # 使用固定內存
_C.TRAIN.LOG_PATH = "./logs/train_log"        # 訓練日誌路徑
_C.TRAIN.SAVE_WEIGHT_PATH = "./logs/weights"  # 權重保存路徑
_C.TRAIN.SAVE_INTERVAL = 5         # 保存間隔（輪次）
_C.TRAIN.SAVE_FREQ = 5             # 每多少個 epoch 保存一次
_C.TRAIN.VAL_FREQ = 1              # 每多少個 epoch 驗證一次
_C.TRAIN.CHECKPOINT = ""           # 檢查點路徑

# 損失函數設置
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = "CrossEntropyLoss"    # 損失函數名稱
_C.TRAIN.LOSS.WEIGHTS = [1.0, 2.0]         # 各類別權重

# 優化器設置
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "Adam"          # 優化器名稱
_C.TRAIN.OPTIMIZER.BETA1 = 0.9            # Adam 的 beta1 參數
_C.TRAIN.OPTIMIZER.BETA2 = 0.999          # Adam 的 beta2 參數

# 學習率調度器設置
_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.NAME = "ReduceLROnPlateau"  # 調度器名稱
_C.TRAIN.SCHEDULER.MODE = "max"               # 調度模式
_C.TRAIN.SCHEDULER.FACTOR = 0.1               # 學習率調整因子
_C.TRAIN.SCHEDULER.PATIENCE = 5               # 容忍輪數
_C.TRAIN.SCHEDULER.VERBOSE = True             # 是否輸出詳細信息

# 測試設置
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 2             # 測試批大小
_C.TEST.NUM_WORKERS = 4            # 測試數據加載進程數
_C.TEST.PIN_MEMORY = True          # 使用固定內存
_C.TEST.CHECKPOINT = ""            # 測試檢查點路徑
_C.TEST.RESULTS_PATH = "./logs/test_result"  # 測試結果路徑

# 對抗攻擊設置
_C.GENERATE_ADV = CN()
_C.GENERATE_ADV.ATTACK_TYPE = "FGSM"          # 攻擊類型：FGSM, PGD, CW, DeepFool
_C.GENERATE_ADV.EPS = 0.1                     # 擾動大小
_C.GENERATE_ADV.ALPHA = 0.01                  # PGD步長
_C.GENERATE_ADV.STEPS = 40                    # PGD步數
_C.GENERATE_ADV.SAVE_PATH = "./adv_samples"   # 對抗樣本保存路徑
_C.GENERATE_ADV.BATCH_SIZE = 16               # 生成批大小
_C.GENERATE_ADV.NUM_WORKERS = 8               # 數據加載進程數
_C.GENERATE_ADV.LOG_INTERVAL = 10             # 日誌間隔
_C.GENERATE_ADV.MEAN = [0.0]                  # 正規化均值
_C.GENERATE_ADV.STD = [1.0]                   # 正規化標準差
_C.GENERATE_ADV.FLIST = "flist_adv.txt"       # 對抗樣本文件列表

def get_cfg_defaults():
    """
    獲取默認配置，調用方法：
    
    # 從文件載入配置
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    
    # 可選：從命令行參數覆蓋
    # cfg.merge_from_list(args.opts)
    
    # 凍結配置
    cfg.freeze()
    """
    return _C.clone() 
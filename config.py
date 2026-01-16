import torch


class Config:
    DATASET_PATH = "./data" # Path to dataset
    IMG_SIZE = 64          
    BATCH_SIZE = 256       
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    LR_DROP_EPOCH = 30
    CELEBA_FILES = {
        "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
        "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
        "identity_CelebA.txt": "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
        "list_bbox_celeba.txt": "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
        "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pd0FJY3Blby1HUTQ"
    }
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TIMESTEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02

    DDIM_STEPS = 75       
    DDIM_ETA = 0.25       

    #Save checkpoints
    SAVE_DIR = "./checkpoints"

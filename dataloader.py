import os
import torchvision.transforms as T
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from config import Config
import gdown

pin_memory=True


def check_and_download_celeba(root):
    """
    Manually downloads CelebA files to satisfy torchvision requirements
    when the automatic download fails.
    """
    # torchvision expects files inside a 'celeba' folder
    base_folder = os.path.join(root, "celeba")
    os.makedirs(base_folder, exist_ok=True)

    print(f"Checking dataset files in {base_folder}...")

    for filename, file_id in Config.CELEBA_FILES.items():
        file_path = os.path.join(base_folder, filename)

        # Only download if it doesn't exist
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"Found {filename} - skipping download.")
def get_dataloader():
    transforms = T.Compose([
        T.Resize(80),
        T.CenterCrop(Config.IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    #download files if not available
    check_and_download_celeba(Config.DATASET_PATH)

    try:
        dataset = CelebA(
        root=Config.DATASET_PATH, 
        split='train', 
        transform=transforms, 
        download=True  # Validates files and unzips automatically.
            )
    except RuntimeError as e:
        print("\nERROR: Torchvision still thinks files are missing.")
        raise e

    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )
    
    return dataloader

if __name__ == "__main__":
    dl = get_dataloader()
    print(f"Dataloader ready. Number of batches: {len(dl)}")
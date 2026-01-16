import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from config import Config
from model import SimpleUNet
from train import get_noise_schedule


@torch.no_grad()
def sample(model_path, num_samples=8):
    device = Config.DEVICE

    # Load model
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Sampling from the model...")

    # Diffusion constants
    betas, alphas, alphas_cumprod = get_noise_schedule()
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    # Start from pure noise
    img = torch.randn(
        (num_samples, 3, Config.IMG_SIZE, Config.IMG_SIZE),
        device=device
    )

    # Reverse diffusion
    for i in reversed(range(Config.TIMESTEPS)):
        t = torch.full(
            (num_samples,),
            i / Config.TIMESTEPS,
            device=device
        )

        noise_pred = model(img, t)

        beta_t = betas[i]
        alpha_t = alphas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]

        model_mean = (1 / torch.sqrt(alpha_t)) * (
            img - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred
        )

        if i > 0:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(beta_t) * noise
        else:
            img = model_mean

    # Visualization (grid)
    img = (img + 1) / 2  # [-1,1] → [0,1]
    img = img.clamp(0, 1)

    grid = vutils.make_grid(img, nrow=4)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title("Generated CelebA Faces")

    output_file = "generated_faces_grid.png"
    plt.savefig(output_file)
    plt.show()

    print(f"Saved generated images to {output_file}")


if __name__ == "__main__":
    latest_model = f"{Config.SAVE_DIR}/ema_model_epoch_{Config.EPOCHS}.pth"
    sample(latest_model, num_samples=8)

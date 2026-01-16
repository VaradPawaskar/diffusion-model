import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from config import Config
from model import SimpleUNet
from train import get_noise_schedule


@torch.no_grad()
def sample_ddim(model_path, num_samples=8):
    device = Config.DEVICE

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(
        f"DDIM sampling | steps={Config.DDIM_STEPS}, "
        f"eta={Config.DDIM_ETA}"
    )

    # Diffusion schedule
    _, _, alphas_cumprod = get_noise_schedule()
    alphas_cumprod = alphas_cumprod.to(device)

    # DDIM timesteps
    step = Config.TIMESTEPS // Config.DDIM_STEPS
    timesteps = list(range(0, Config.TIMESTEPS, step))[::-1]

    img = torch.randn(
        (num_samples, 3, Config.IMG_SIZE, Config.IMG_SIZE),
        device=device
    )

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        t_tensor = torch.full(
            (num_samples,),
            t / Config.TIMESTEPS,
            device=device
        )

        eps = model(img, t_tensor)

        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next]

        # Predict x0
        x0_pred = (img - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

        # CLAMP 
        x0_pred = x0_pred.clamp(-1, 1)

        # Stochastic DDIM
        eta = Config.DDIM_ETA
        sigma = eta * torch.sqrt(
            (1 - alpha_next) / (1 - alpha_t) *
            (1 - alpha_t / alpha_next)
        )

        noise = torch.randn_like(img)

        img = (
            torch.sqrt(alpha_next) * x0_pred +
            torch.sqrt(1 - alpha_next - sigma**2) * eps +
            sigma * noise
        )

    # Visualization
    img = (img + 1) / 2
    img = img.clamp(0, 1)

    grid = vutils.make_grid(img, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title("CelebA Faces – DDIM (Refined)")
    plt.savefig("final_ddim_faces.png")
    plt.show()


if __name__ == "__main__":
    latest = f"{Config.SAVE_DIR}/ema_model_epoch_{Config.EPOCHS}.pth"
    sample_ddim(latest, num_samples=8)

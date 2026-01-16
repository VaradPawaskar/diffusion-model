import torch
import torch.nn as nn
import os
from tqdm import tqdm

from config import Config
from dataloader import get_dataloader
from model import SimpleUNet

torch.backends.cudnn.benchmark = True

#Diffusion Schedule
def get_noise_schedule():
    betas = torch.linspace(
        Config.BETA_START,
        Config.BETA_END,
        Config.TIMESTEPS
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

# Forward Diffusion 
def forward_diffusion_sample(x0, t, alphas_cumprod):
    noise = torch.randn_like(x0)

    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]

    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
    return xt, noise


# EMA Helper Class
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.model = model

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_((1.0 - self.decay) * param.data)

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

# Training Loop
def train():
    device = Config.DEVICE
    print(f"Training on {device}")

    dataloader = get_dataloader()
    model = SimpleUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[Config.LR_DROP_EPOCH],
        gamma=0.5
    )

    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    betas, alphas, alphas_cumprod = get_noise_schedule()
    alphas_cumprod = alphas_cumprod.to(device)

    ema = EMA(model, decay=0.999)

    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    #Epoch Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        pbar = tqdm(dataloader)

        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.size(0)

            t = torch.randint(
                0, Config.TIMESTEPS,
                (batch_size,),
                device=device
            )

            x_t, noise = forward_diffusion_sample(
                images, t, alphas_cumprod
            )

            t_norm = t.float() / Config.TIMESTEPS

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                noise_pred = model(x_t, t_norm)
                loss = criterion(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update()

            pbar.set_description(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Loss {loss.item():.4f}"
            )

        scheduler.step()

        # Save EMA checkpoint
        ema.apply_shadow()
        ckpt_path = f"{Config.SAVE_DIR}/ema_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        ema.restore()

        print(f"Saved EMA checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()

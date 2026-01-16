Diffusion Models from Scratch: Implement a minimal diffusion model
Author: Varad Satish Pawaskar
Date: 01.01.2026

1. PROJECT OVERVIEW
-------------------
This repository contains a PyTorch implementation of a Diffusion model
trained on the CelebA dataset. It includes a 
custom U-Net architecture with Self-Attention mechanisms and supports both 
standard DDPM sampling and accelerated DDIM (Denoising Diffusion Implicit 
Models) sampling.

Key technical features include:
- Exponential Moving Average (EMA) for stable model weight updates.
- Mixed Precision Training (FP16) via torch.cuda.amp.
- Sinusoidal Positional Embeddings.
- U-Net with Residual Blocks and Group Normalization.
- Configurable inference using DDIM for faster generation.

2. PROJECT STRUCTURE
--------------------
- config.py         : Central configuration (hyperparameters, paths, device).
- dataloader.py     : Handles CelebA dataset downloading and preprocessing.
- model.py          : Defines the U-Net architecture, ResBlocks, and Attention.
- train.py          : Main training loop with EMA and Mixed Precision.
- evaluate_ddpm.py  : Standard stochastic sampling (slow, high quality).
- evaluate_ddim.py  : Stochastic accelerated sampling.
- visualization.py  : Helper functions for tensor-to-image conversion.

3. REQUIREMENTS
---------------
Ensure you have a Python (Version 3.10) environment set up with the following dependencies:

    pip install -r requirements.txt

Note: CUDA is highly recommended for training. The code automatically detects
available hardware (defined in config.py).

4. DATASET
----------
The model is trained on the CelebA dataset.
- The `dataloader.py` script automatically downloads the dataset if it is 
  not present in the `./data` directory.
- Images are resized to 80x80 and center-cropped to 64x64.
- Pixel values are normalized to the range [-1, 1].

5. CONFIGURATION
----------------
All hyperparameters are managed in `config.py`.

6. USAGE
--------

A. Training
   To train the model from scratch:
   
       python train.py

   - This will create a `./data` folder (if missing) and download CelebA.
   - Checkpoints are saved to `./checkpoints`.
   - The script uses EMA (decay=0.999) and saves the EMA model at every epoch.

B. Inference (Standard DDPM)
   To generate images using the full 1000-step reverse diffusion process:
   
       python evaluate_ddpm.py

   - Loads the model from the final epoch by default.
   - Saves the output as `generated_faces_grid.png`.

C. Inference (Accelerated DDIM)
   To generate images faster with controllable stochasticity:
   
       python evaluate_ddim.py

   - Uses the configuration `DDIM_STEPS` and `DDIM_ETA` from config.py.
   - Saves the output as `final_ddim_faces.png`.


==============================================================================
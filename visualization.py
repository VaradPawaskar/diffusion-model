import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

def show_tensor_image(image):
    reverse_transforms = T.Compose([
        T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        T.ToPILImage()
    ])

    if len(image.shape) == 4:
        image = image[0]

    image = image.clamp(-1, 1)

    plt.imshow(reverse_transforms(image), interpolation="nearest")
    plt.axis("off")

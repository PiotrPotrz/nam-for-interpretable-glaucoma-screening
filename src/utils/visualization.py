import wandb
import matplotlib.pyplot as plt
import torch

def plot_sample(image, mask, prediction):
    fig, ax = plt.subplots(1, 3)

    prediction = torch.argmax(prediction, dim=0)

    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Image")

    ax[1].imshow(mask.cpu().numpy())
    ax[1].set_title("Mask")

    ax[2].imshow(prediction.cpu().numpy())
    ax[2].set_title("Prediction")

    plt.tight_layout()

    wandb.log({"viz": wandb.Image(plt)})
    plt.close()
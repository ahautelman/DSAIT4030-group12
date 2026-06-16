import os
import sys
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

if platform == "linux" or platform == "linux2":
    # We assume that the project folder is located in the home directory
    home_dir = os.path.expanduser("~")
    sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'DSAIT4030-group12')))

from diffuser.diffuser_example.toy_image_generator import get_toy_image_example_batch
from diffuser.diffuser_ddpm_linear_schedule import Diffuser_DDPM_linear_schedule
from diffuser.unet import DiffusionUNet
from diffuser.diffuser_example.unet_example_config import DiffuserConfig

def make_toy_batch(device, batch_size=8, x_dim=64, y_dim=64):
    # Hue values are in [0, 180], so we pick a starting hue and sample all other values based on the batch size.
    start_hue_1s = np.random.randint(0, 180)
    hue_1_offsets = np.linspace(0, 180, num=batch_size, endpoint=False)
    hue_1s = (start_hue_1s + hue_1_offsets) % 180

    # Hue 2 is always the "opposite" hue, i.e. 90 degrees apart in the hue circle.
    hue_2s = (hue_1s + 90) % 180

    # Maximize saturation and value for the coloured regions
    saturation_1s = np.full(batch_size, 255)
    saturation_2s = np.full(batch_size, 255)
    value_1s = np.full(batch_size, 255)
    value_2s = np.full(batch_size, 255)
    
    # The line remains black
    hue_3s = np.zeros(batch_size)
    saturation_3s = np.zeros(batch_size)
    value_3s = np.zeros(batch_size)

    # Angles are in [0, 360]
    start_angle = np.random.rand() * 360
    angle_offsets = np.linspace(0, 360, num=batch_size, endpoint=False)
    angles = (start_angle + angle_offsets) % 360
    
    # Line width
    line_widths = np.random.randint(16, 32, size=batch_size)

    # Retrieve batch
    batch = get_toy_image_example_batch(
        x_dim=x_dim,
        y_dim=y_dim,
        hue_1s=hue_1s,
        saturation_1s=saturation_1s,
        value_1s=value_1s,
        hue_2s=hue_2s,
        saturation_2s=saturation_2s,
        value_2s=value_2s,
        hue_3s=hue_3s,
        saturation_3s=saturation_3s,
        value_3s=value_3s,
        angles=angles,
        line_widths=line_widths,
    )
    return batch.to(device)

def train_step(data, minibatch_size, model, ddpm, optimizer, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    number_of_minibatches = data.shape[0] // minibatch_size
    total_loss = 0.0

    for i in range(number_of_minibatches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        data_minibatch = data[start_idx:end_idx]

        B = data_minibatch.shape[0]
        t = torch.randint(0, ddpm.total_timesteps, (B,), device=device, dtype=torch.long).view(-1)

        x_t, true_noise = ddpm.forward_diffusion(data_minibatch, t)
        pred_noise = model(x_t, t)
        
        loss = F.mse_loss(pred_noise, true_noise) / number_of_minibatches
        loss.backward()
        total_loss += loss.detach().item()
        
    optimizer.step()
    return total_loss

@torch.no_grad()
def sample(model, ddpm, shape, device, fixed_noise=None):
    model.eval()
    x = torch.randn(shape, device=device) if fixed_noise is None else fixed_noise

    for t in range(ddpm.total_timesteps - 1, 0, -1):
        ts = torch.full((shape[0],), t, device=device, dtype=torch.long).view(-1)
        pred_noise = model(x, ts)
        x = ddpm.reverse_diffusion(x, ts, pred_noise)

    return x

def show_images(img_batch, title=None, save_path=None):
    img_batch = img_batch.detach().cpu()
    fig, axes = plt.subplots(1, img_batch.shape[0], figsize=(4 * img_batch.shape[0], 4))
    if img_batch.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = img_batch[i]
        img = (img + 1.0) / 2.0
        img = (img * 255.0).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

    if title:
        fig.suptitle(title)
        
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def save_checkpoint(step, model, optimizer, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"unet_checkpoint_step_{step}.pt")
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)

def main():
    num_steps = 20000
    batch_size = 512
    minibatch_size = 128
    print_loss_every = 25
    preview_every = 200
    checkpoint_every = 1000  
    
    os.makedirs("previews", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)

    ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    ddpm_model.betas = ddpm_model.betas.to(device)
    ddpm_model.alphas = ddpm_model.alphas.to(device)
    ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

    config = DiffuserConfig()
    unet = DiffusionUNet(
        config=config,
        model_in_channels=3,
        model_out_channels=3
    ).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4, weight_decay=1e-5)

    loss_history = []
    start_step = 0
    
    # 3. Training Loop
    if start_step < num_steps:
        print("Starting training...")
    else:
        print("Training already completed up to num_steps.")
        
    for step in range(start_step, num_steps):
        data = make_toy_batch(device=device, batch_size=batch_size, x_dim=64, y_dim=64)
        
        loss = train_step(data, minibatch_size, unet, ddpm_model, optimizer, device)
        loss_history.append(loss)

        if step % print_loss_every == 0:
            print(f"step {step:5d} | loss {loss:.4f}")

        if step > 0 and step % preview_every == 0:
            samples = sample(unet, ddpm_model, (1, 3, 64, 64), device)
            preview_path = f"previews/preview_step_{step}.png"
            show_images(samples, title=f"step {step}", save_path=preview_path) 
            unet.train()
            
        # Checkpoint Saving
        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(step, unet, optimizer, loss)

    # Final Checkpoint and Sample (only if we actually ran some steps)
    if start_step < num_steps:
        save_checkpoint(num_steps, unet, optimizer, loss_history[-1] if loss_history else 0)
        samples = sample(unet, ddpm_model, (4, 3, 64, 64), device)
        
        # Save the final generation
        final_preview_path = "previews/preview_final.png"
        show_images(samples, title="Generated samples (Final)", save_path=final_preview_path)
        print(f"--> Training complete! Final preview saved to {final_preview_path}")

if __name__ == "__main__":
    main()
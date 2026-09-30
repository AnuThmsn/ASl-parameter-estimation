import sys, os, time, numpy as np, torch, yaml
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import generate_spatial_volume
from src.physics.noise_models import add_noise

with open("experiments/literature_unet/configs/unet_config.yaml") as f:
    cfg = yaml.safe_load(f)

D=cfg["data"]["D"]; H=cfg["data"]["H"]; W=cfg["data"]["W"]
rng = np.random.default_rng(999)
sig, cbf, att, mask = generate_spatial_volume(rng, D, H, W, noise_sd=0.0)

norm = np.load("experiments/literature_unet/checkpoints/unet_norm_seed42.npz")
sig_mean=norm["sig_mean"]; sig_std=norm["sig_std"]
cbf_mean=float(norm["cbf_mean"]); cbf_std=float(norm["cbf_std"])
att_mean=float(norm["att_mean"]); att_std=float(norm["att_std"])

model = LuciwUNet3D(4, cfg["model"]["base_channels"])
model.load_state_dict(torch.load("experiments/literature_unet/checkpoints/unet_4pld_seed42.pt", weights_only=True))
model.eval()

SNRs = [np.inf, 20, 10, 5]
os.makedirs("experiments/literature_unet/figures", exist_ok=True)

for snr in SNRs:
    if np.isinf(snr):
        noisy_sig = sig.copy()
    else:
        nsd = 334.2039 / snr
        flat = sig.transpose(1,2,3,0).reshape(-1,4)
        noisy_flat = add_noise(flat, nsd, rng=np.random.default_rng(123))
        noisy_sig = noisy_flat.reshape(D,H,W,4).transpose(3,0,1,2)

    sig_norm = (noisy_sig - sig_mean) / sig_std
    x_t = torch.tensor(sig_norm[None], dtype=torch.float32)
    with torch.no_grad():
        out = model(x_t)
    
    cbf_pred = out[0,0].numpy() * cbf_std + cbf_mean
    att_pred = out[0,1].numpy() * att_std + att_mean
    
    # Masking for visual clarity
    cbf_pred = np.where(mask > 0.5, cbf_pred, 0)
    att_pred = np.where(mask > 0.5, att_pred, 0)
    
    z = D // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"U-Net Predictions (SNR={snr})", fontsize=16)
    
    # PLDs
    for i in range(4):
        ax = axes[0, i]
        im = ax.imshow(noisy_sig[i, z, :, :], cmap='gray')
        ax.set_title(f"PLD {i+1} Input")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    # Targets
    ax = axes[1, 0]
    im = ax.imshow(cbf[z, :, :], cmap='hot', vmin=0, vmax=100)
    ax.set_title("True CBF")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 1]
    im = ax.imshow(cbf_pred[z, :, :], cmap='hot', vmin=0, vmax=100)
    ax.set_title("Predicted CBF")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 2]
    im = ax.imshow(att[z, :, :], cmap='viridis', vmin=0.5, vmax=3.0)
    ax.set_title("True ATT")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 3]
    im = ax.imshow(att_pred[z, :, :], cmap='viridis', vmin=0.5, vmax=3.0)
    ax.set_title("Predicted ATT")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"experiments/literature_unet/figures/prediction_snr_{snr}.png", dpi=150)
    plt.close()

print("Figures generated.")

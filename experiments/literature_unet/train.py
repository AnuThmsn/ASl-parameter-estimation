"""
train.py - Standard 4-PLD U-Net training.
Inspired by Luciw et al. 2022 (DOI: 10.1002/mrm.29193).
Adapted for synthetic 4-PLD ASL data with 32x32x16 volumes.
NOTE: Spatial dimensions reduced from 64x64 to 32x32 due to CPU compute
constraints. This is documented as a hardware adaptation; scientific
conclusions hold at the smaller scale.
"""
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import generate_spatial_volume


def masked_mae_loss(pred, target, mask, brain_weight=1.0, bg_weight=1.0):
    brain = (mask > 0.5)
    bg = ~brain
    loss = 0.0; n_terms = 0
    if brain.any():
        loss += brain_weight * (pred[brain] - target[brain]).abs().mean(); n_terms += 1
    if bg.any():
        loss += bg_weight * (pred[bg] - target[bg]).abs().mean(); n_terms += 1
    return loss / max(n_terms, 1)


def train_one_seed(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    D = cfg["data"]["D"]; H = cfg["data"]["H"]; W = cfg["data"]["W"]
    n_train = cfg["data"]["n_train"]; n_val = cfg["data"]["n_val"]
    sd_levels = np.linspace(0.0, 334.2039 / 5.0, 100)

    print(f"\n=== Seed {seed}: Generating {n_train} training volumes ===", flush=True)
    rng_tr = np.random.default_rng(cfg["data"]["train_seed"] + seed)
    train_sigs, train_cbf, train_att, train_masks = [], [], [], []
    for i in range(n_train):
        nd = rng_tr.choice(sd_levels)
        sig, cbf, att, mask = generate_spatial_volume(rng_tr, D, H, W, nd)
        train_sigs.append(sig); train_cbf.append(cbf[None])
        train_att.append(att[None]); train_masks.append(mask[None].astype(np.float32))

    train_sigs = np.array(train_sigs, dtype=np.float32)
    train_cbf  = np.array(train_cbf,  dtype=np.float32)
    train_att  = np.array(train_att,  dtype=np.float32)
    train_masks= np.array(train_masks,dtype=np.float32)

    # Normalization from training data ONLY
    sig_mean = train_sigs.mean(axis=(0,2,3,4), keepdims=True)[0]
    sig_std  = train_sigs.std( axis=(0,2,3,4), keepdims=True)[0]
    sig_std[sig_std == 0] = 1.0
    bidx = train_masks[:,0] > 0.5
    cbf_brain = train_cbf[:,0][bidx]; att_brain = train_att[:,0][bidx]
    cbf_mean = float(cbf_brain.mean()); cbf_std  = float(cbf_brain.std()) + 1e-8
    att_mean = float(att_brain.mean()); att_std  = float(att_brain.std()) + 1e-8
    print(f"  CBF norm: mean={cbf_mean:.1f} std={cbf_std:.1f}  ATT: mean={att_mean:.3f} std={att_std:.3f}", flush=True)

    os.makedirs("experiments/literature_unet/checkpoints", exist_ok=True)
    np.savez(f"experiments/literature_unet/checkpoints/unet_norm_seed{seed}.npz",
             sig_mean=sig_mean, sig_std=sig_std, cbf_mean=cbf_mean,
             cbf_std=cbf_std, att_mean=att_mean, att_std=att_std)

    tr_sn=(train_sigs-sig_mean)/sig_std; tr_cn=(train_cbf-cbf_mean)/cbf_std; tr_an=(train_att-att_mean)/att_std

    print(f"  Generating {n_val} val volumes...", flush=True)
    rng_va = np.random.default_rng(cfg["data"]["val_seed"])
    val_sigs, val_cbf, val_att, val_masks = [], [], [], []
    for i in range(n_val):
        nd = rng_va.choice(sd_levels)
        sig, cbf, att, mask = generate_spatial_volume(rng_va, D, H, W, nd)
        val_sigs.append(sig); val_cbf.append(cbf[None])
        val_att.append(att[None]); val_masks.append(mask[None].astype(np.float32))
    val_sigs = np.array(val_sigs, dtype=np.float32)
    val_cbf  = np.array(val_cbf,  dtype=np.float32)
    val_att  = np.array(val_att,  dtype=np.float32)
    val_masks= np.array(val_masks,dtype=np.float32)
    va_sn=(val_sigs-sig_mean)/sig_std; va_cn=(val_cbf-cbf_mean)/cbf_std; va_an=(val_att-att_mean)/att_std

    bs = cfg["training"]["batch_size"]
    train_dl = DataLoader(TensorDataset(torch.tensor(tr_sn), torch.tensor(tr_cn),
        torch.tensor(tr_an), torch.tensor(train_masks)), batch_size=bs, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(va_sn), torch.tensor(va_cn),
        torch.tensor(va_an), torch.tensor(val_masks)), batch_size=bs, shuffle=False)

    model = LuciwUNet3D(in_channels=4, base_channels=cfg["model"]["base_channels"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}", flush=True)

    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[cfg["training"]["lr_step_epoch"]], gamma=cfg["training"]["lr_step_factor"])
    bw=cfg["loss"]["brain_weight"]; bgw=cfg["loss"]["background_weight"]
    patience=cfg["training"]["patience"]; epochs=cfg["training"]["epochs"]
    grad_clip=cfg["training"]["grad_clip"]

    best_val=float("inf"); best_weights=None; p_ctr=0; t0=time.time()
    for ep in range(epochs):
        model.train(); tr_loss=0.0
        for sn,cn,an,mn in train_dl:
            optimizer.zero_grad()
            out=model(sn)
            loss=masked_mae_loss(out[:,0:1],cn,mn,bw,bgw)+masked_mae_loss(out[:,1:2],an,mn,bw,bgw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); optimizer.step()
            tr_loss+=loss.item()*sn.size(0)
        tr_loss/=len(train_dl.dataset)
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for sn,cn,an,mn in val_dl:
                out=model(sn)
                loss=masked_mae_loss(out[:,0:1],cn,mn,bw,bgw)+masked_mae_loss(out[:,1:2],an,mn,bw,bgw)
                va_loss+=loss.item()*sn.size(0)
        va_loss/=len(val_dl.dataset)
        scheduler.step()
        if va_loss<best_val:
            best_val=va_loss; best_weights={k:v.cpu().clone() for k,v in model.state_dict().items()}; p_ctr=0
        else:
            p_ctr+=1
            if p_ctr>=patience: print(f"  Early stop ep {ep}", flush=True); break
        if ep%5==0 or ep<3:
            print(f"  Ep{ep:3d} tr={tr_loss:.4f} val={va_loss:.4f} best={best_val:.4f} {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_weights)
    ckpt=f"experiments/literature_unet/checkpoints/unet_4pld_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved: {ckpt}  time={time.time()-t0:.0f}s", flush=True)
    return cbf_mean, cbf_std, att_mean, att_std, sig_mean, sig_std


def main():
    with open("experiments/literature_unet/configs/unet_config.yaml") as f:
        cfg = yaml.safe_load(f)
    for seed in cfg["training"]["seeds"]:
        train_one_seed(cfg, seed)
    print("\nAll seeds complete.", flush=True)


if __name__ == "__main__":
    main()

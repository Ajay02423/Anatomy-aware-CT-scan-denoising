# training_baseline_fixed.py
import os, glob, random, argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# ============================================================
# ------------------------- Utils ----------------------------
# ============================================================

def set_seed(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(img_np):
    # Expects input in [0, 1], scales to [-1, 1] for Tanh
    x = img_np * 2.0 - 1.0
    return torch.from_numpy(x)[None, ...].float()

@torch.no_grad()
def save_png(x, path):
    x = (x.clamp(-1,1) + 1)/2
    x = (x.squeeze(0).cpu().numpy()*255.0).astype(np.uint8)
    Image.fromarray(x).save(path)

# ============================================================
# ------------------------- SSIM -----------------------------
# ============================================================

class SSIM(nn.Module):
    def __init__(self, win=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(win).float() - (win-1)/2
        g = torch.exp(-(coords**2)/(2*sigma**2)); g /= g.sum()
        w = (g[:,None] @ g[None,:])[None,None,:,:]
        self.register_buffer('w', w)
        self.C1 = 0.01**2
        self.C2 = 0.03**2
    def forward(self, x, y):
        w = self.w.to(x.dtype)
        mu_x = F.conv2d(x, w, padding=w.shape[-1]//2, groups=1)
        mu_y = F.conv2d(y, w, padding=w.shape[-1]//2, groups=1)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sig_x2 = F.conv2d(x*x, w, padding=w.shape[-1]//2, groups=1) - mu_x2
        sig_y2 = F.conv2d(y*y, w, padding=w.shape[-1]//2, groups=1) - mu_y2
        sig_xy = F.conv2d(x*y, w, padding=w.shape[-1]//2, groups=1) - mu_xy
        num = (2*mu_xy + self.C1) * (2*sig_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sig_x2 + sig_y2 + self.C2)
        return (num / (den + 1e-8)).clamp(0,1).mean()

# ============================================================
# ------------------------- Data -----------------------------
# ============================================================

DOSES_ALLOWED = [10,25,50,70]

def find_lodo_samples(root):
    # UPDATED: Prioritize _hu.npy as requested
    roots = sorted(glob.glob(os.path.join(root, "sample_*")))
    out = []
    for r in roots:
        # Check for NDCT_hu.npy
        nd = os.path.join(r, "NDCT_hu.npy")
        if not os.path.isfile(nd): 
            # Fallback if needed, but primary is hu
            nd_alt = os.path.join(r, "NDCT_mu.npy")
            if os.path.isfile(nd_alt): nd = nd_alt
            else: continue
            
        doses = []
        for d in DOSES_ALLOWED:
            fp = os.path.join(r, f"LDCT_{d}_hu.npy")
            if not os.path.isfile(fp):
                fp_alt = os.path.join(r, f"LDCT_{d}_mu.npy")
                if os.path.isfile(fp_alt): fp = fp_alt
            if os.path.isfile(fp): doses.append((d, fp))
        if doses:
            out.append(dict(nd=nd, doses=doses))
    return out

def find_mayo_samples(root):
    return find_lodo_samples(root)

class PairDataset(Dataset):
    def __init__(self, samples, pick_random_dose=True, fixed_dose=None):
        self.samples = samples
        self.pick_random_dose = pick_random_dose
        self.fixed_dose = fixed_dose
        # Fixed Normalization Constants (Standard Abdominal/Chest range approx)
        self.MIN_HU = -1000.0
        self.MAX_HU = 1000.0

    def __len__(self): return len(self.samples)

    def normalize(self, img):
        # UPDATED: Fixed Windowing instead of Percentile
        # Clip to standard range and normalize to [0, 1]
        img = np.clip(img, self.MIN_HU, self.MAX_HU)
        img = (img - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        nd = np.load(item['nd']).astype(np.float32)
        
        if self.fixed_dose is not None:
            choices = [p for d,p in item['doses'] if d==self.fixed_dose]
            if not choices: d,p = random.choice(item['doses'])
            else: p = choices[0]; d = self.fixed_dose
        else:
            d,p = random.choice(item['doses'])
            
        ld = np.load(p).astype(np.float32)

        # UPDATED: Use fixed normalization
        ndw = self.normalize(nd)
        ldw = self.normalize(ld)

        # Convert to tensor [-1, 1]
        nd_t = to_tensor(ndw)
        ld_t = to_tensor(ldw)
        
        # UPDATED: Removed resize_tensor()
        
        return nd_t, ld_t, d

# ============================================================
# ------------------------- Models ---------------------------
# ============================================================

def conv_block(in_ch, out_ch, s=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def deconv_block(in_ch, out_ch, s=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=s, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self, in_ch=1, base=64, latent_ch=512):
        super().__init__()
        self.enc = nn.Sequential(
            conv_block(in_ch, base),
            conv_block(base, base),
            nn.MaxPool2d(2),
            conv_block(base, base*2),
            conv_block(base*2, base*2),
            nn.MaxPool2d(2),
            conv_block(base*2, base*4),
            conv_block(base*4, base*4),
            nn.MaxPool2d(2),
            conv_block(base*4, base*8),
            conv_block(base*8, base*8),
            nn.MaxPool2d(2),
            conv_block(base*8, latent_ch),
        )
    def forward(self, x): return self.enc(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, base=64, latent_ch=512):
        super().__init__()
        self.dec = nn.Sequential(
            deconv_block(latent_ch, base*8),
            conv_block(base*8, base*8),
            deconv_block(base*8, base*4),
            conv_block(base*4, base*4),
            deconv_block(base*4, base*2),
            conv_block(base*2, base*2),
            deconv_block(base*2, base),
            conv_block(base, base),
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, z): return self.dec(z)

# ============================================================
# ------------------------- Training -------------------------
# ============================================================

def match_size(pred, target):
    if pred.shape[-1] != target.shape[-1] or pred.shape[-2] != target.shape[-2]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, target

@torch.no_grad()
def save_sample(pred, target, path):
    pred = (pred.clamp(-1,1) + 1)/2
    target = (target.clamp(-1,1) + 1)/2
    grid = make_grid([pred[0], target[0]], nrow=2)
    ndarr = (grid.cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
    Image.fromarray(ndarr).save(path)

# Evaluation helpers (per-epoch)
@torch.no_grad()
def evaluate_epoch_teacher(E, D, val_loader, device, ssim_metric):
    E.eval(); D.eval()
    psnrs, ssims, rmses, mses = [], [], [], []
    for nd, _ in val_loader:
        nd = nd.to(device)
        pred = D(E(nd))
        pred, nd = match_size(pred, nd)
        pred01 = (pred.clamp(-1,1)+1)/2
        nd01 = (nd.clamp(-1,1)+1)/2
        mse = F.mse_loss(pred01, nd01, reduction='mean').item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-12))
        ssim_val = ssim_metric(pred01, nd01).item()
        rmse = math.sqrt(mse)
        psnrs.append(psnr); ssims.append(ssim_val); rmses.append(rmse); mses.append(mse)
    return np.mean(psnrs) if psnrs else float('nan'), np.mean(mses) if mses else float('nan'), np.mean(ssims) if ssims else float('nan'), np.mean(rmses) if rmses else float('nan')

@torch.no_grad()
def evaluate_epoch_student(Es, D, val_loader, device, ssim_metric):
    Es.eval(); D.eval()
    psnrs, ssims, rmses, mses = [], [], [], []
    for nd, ld, _ in val_loader:
        nd, ld = nd.to(device), ld.to(device)
        pred = D(Es(ld))
        pred, nd = match_size(pred, nd)
        pred01 = (pred.clamp(-1,1)+1)/2
        nd01 = (nd.clamp(-1,1)+1)/2
        mse = F.mse_loss(pred01, nd01, reduction='mean').item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-12))
        ssim_val = ssim_metric(pred01, nd01).item()
        rmse = math.sqrt(mse)
        psnrs.append(psnr); ssims.append(ssim_val); rmses.append(rmse); mses.append(mse)
    return np.mean(psnrs) if psnrs else float('nan'), np.mean(mses) if mses else float('nan'), np.mean(ssims) if ssims else float('nan'), np.mean(rmses) if rmses else float('nan')

# ---------------- Student per-dose evaluation (NEW) ----------------
@torch.no_grad()
def evaluate_student_per_dose(Es, D, test_samples, device, batch_size=8):
    Es.eval(); D.eval()
    ssim_metric = SSIM().to(device)
    per_dose = {}
    for d in DOSES_ALLOWED:
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=d)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        psnrs, ssims, rmses, mses = [], [], [], []
        for nd, ld, _ in loader:
            nd, ld = nd.to(device), ld.to(device)
            pred = D(Es(ld))
            pred, nd = match_size(pred, nd)
            pred01 = (pred.clamp(-1,1)+1)/2; nd01 = (nd.clamp(-1,1)+1)/2
            mse = F.mse_loss(pred01, nd01, reduction='mean').item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-12))
            ssim_val = ssim_metric(pred01, nd01).item()
            rmse = math.sqrt(mse)
            psnrs.append(psnr); ssims.append(ssim_val); rmses.append(rmse); mses.append(mse)
        per_dose[d] = {
            "psnr": float(np.mean(psnrs)) if psnrs else float('nan'),
            "mse": float(np.mean(mses)) if mses else float('nan'),
            "ssim": float(np.mean(ssims)) if ssims else float('nan'),
            "rmse": float(np.mean(rmses)) if rmses else float('nan'),
            "n_samples": len(loader.dataset)
        }
    return per_dose

# ---------------- Teacher ----------------
def train_teacher(loader, device, out_dir, epochs=50, lr=2e-4, alpha_ssim=0.2, val_loader=None):
    E, D = Encoder().to(device), Decoder().to(device)
    opt = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=lr)
    ssim = SSIM().to(device); l1 = nn.L1Loss()
    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    step = 0

    for ep in range(1, epochs + 1):
        E.train(); D.train(); total = 0.0
        for nd, _ in loader:
            nd = nd.to(device)
            pred = D(E(nd))
            pred, nd = match_size(pred, nd)
            ssim_val = ssim(pred, nd)
            loss = l1(pred, nd) + alpha_ssim * (1 - ssim_val)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); writer.add_scalar("teacher/train_loss", loss.item(), step); step += 1

        avg_train_loss = total/len(loader) if len(loader)>0 else 0.0
        print(f"[Teacher] Epoch {ep}/{epochs} loss={avg_train_loss:.4f}")

        with torch.no_grad():
            try:
                save_sample(D(E(nd[:1])), nd[:1], os.path.join(out_dir, "samples", f"teacher_ep{ep}.png"))
            except Exception:
                pass

        if val_loader is not None:
            psnr, mse, ssim_val, rmse = evaluate_epoch_teacher(E, D, val_loader, device, ssim)
            print(f"[Teacher][Val] Epoch {ep}: PSNR={psnr:.3f}, MSE={mse:.6e}, SSIM={ssim_val:.4f}, RMSE={rmse:.5f}")
            writer.add_scalar("teacher/val_psnr", psnr, ep)
            writer.add_scalar("teacher/val_mse", mse, ep)
            writer.add_scalar("teacher/val_ssim", ssim_val, ep)
            writer.add_scalar("teacher/val_rmse", rmse, ep)

    torch.save(E.state_dict(), os.path.join(out_dir, "teacher_Ec.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "decoder_D.pt"))
    writer.close(); return E, D

# ---------------- Student ----------------
def train_student(loader, device, out_dir, Ec, D, epochs=50, lr=2e-4, lam_lat=1.0, lam_rec=1.0, val_loader=None, test_samples=None):
    for p in Ec.parameters(): p.requires_grad=False
    for p in D.parameters(): p.requires_grad=False
    Es = Encoder().to(device); opt = torch.optim.Adam(Es.parameters(), lr=lr)
    l1 = nn.L1Loss(); writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True); step = 0

    ssim = SSIM().to(device)

    for ep in range(1, epochs + 1):
        Es.train(); total = 0.0
        for nd, ld, _ in loader:
            nd, ld = nd.to(device), ld.to(device)
            with torch.no_grad(): zt = Ec(nd)
            zs = Es(ld); pred = D(zs); pred, nd = match_size(pred, nd)
            loss_lat = F.mse_loss(zs, zt); loss_rec = l1(pred, nd)
            loss = lam_lat * loss_lat + lam_rec * loss_rec
            opt.zero_grad(); loss.backward(); opt.step()
            writer.add_scalar("student/total_loss", loss.item(), step); total += loss.item(); step += 1
        avg_train_loss = total/len(loader) if len(loader)>0 else 0.0
        print(f"[Student] Epoch {ep}/{epochs} loss={avg_train_loss:.4f}")

        with torch.no_grad():
            try:
                save_sample(D(Es(ld[:1])), nd[:1], os.path.join(out_dir, "samples", f"student_ep{ep}.png"))
            except Exception:
                pass

        # Per-epoch overall evaluation
        if val_loader is not None:
            psnr, mse, ssim_val, rmse = evaluate_epoch_student(Es, D, val_loader, device, ssim)
            print(f"[Student][Val] Epoch {ep}: PSNR={psnr:.3f}, MSE={mse:.6e}, SSIM={ssim_val:.4f}, RMSE={rmse:.5f}")
            writer.add_scalar("student/val_psnr", psnr, ep)
            writer.add_scalar("student/val_mse", mse, ep)
            writer.add_scalar("student/val_ssim", ssim_val, ep)
            writer.add_scalar("student/val_rmse", rmse, ep)

        # Per-dose evaluation (NEW) — requires test_samples
        if test_samples is not None:
            per_dose = evaluate_student_per_dose(Es, D, test_samples, device, batch_size=8)
            print(f"[Student][Per-dose][Epoch {ep}]")
            for d in sorted(per_dose.keys()):
                info = per_dose[d]
                print(f"  Dose {d}% | n={info['n_samples']:4d} | PSNR={info['psnr']:.3f} | MSE={info['mse']:.6e} | SSIM={info['ssim']:.4f} | RMSE={info['rmse']:.5f}")
                writer.add_scalar(f"student/perdose/{d}_psnr", info['psnr'], ep)
                writer.add_scalar(f"student/perdose/{d}_mse", info['mse'], ep)
                writer.add_scalar(f"student/perdose/{d}_ssim", info['ssim'], ep)
                writer.add_scalar(f"student/perdose/{d}_rmse", info['rmse'], ep)

    torch.save(Es.state_dict(), os.path.join(out_dir, "student_Es.pt")); writer.close(); return Es

# ============================================================
# 🔹 Dose-wise Evaluation (PSNR, SSIM, RMSE) (keeps existing)
# ============================================================

@torch.no_grad()
def evaluate_model_dosewise(E, D, test_samples, device, out_dir="eval_results"):
    os.makedirs(out_dir, exist_ok=True)
    E.eval(); D.eval(); ssim_metric = SSIM().to(device)
    dose_metrics = {d: {"psnr": [], "ssim": [], "rmse": []} for d in DOSES_ALLOWED}

    for d in DOSES_ALLOWED:
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=d)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        print(f"[Eval Dose {d}%] {len(loader.dataset)} samples")

        for nd, ld, _ in tqdm(loader, desc=f"Dose {d}%"):
            nd, ld = nd.to(device), ld.to(device)
            pred = D(E(ld)); pred, nd = match_size(pred, nd)
            pred01, nd01 = (pred.clamp(-1,1)+1)/2, (nd.clamp(-1,1)+1)/2
            mse = F.mse_loss(pred01, nd01).item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-12))
            ssim_val = ssim_metric(pred01, nd01).item()
            rmse = math.sqrt(mse)
            dose_metrics[d]["psnr"].append(psnr)
            dose_metrics[d]["ssim"].append(ssim_val)
            dose_metrics[d]["rmse"].append(rmse)

        save_sample(pred[:1], nd[:1], os.path.join(out_dir, f"dose_{d}_example.png"))

    print("\n===== Dose-wise Metrics =====")
    for d in DOSES_ALLOWED:
        psnr = np.mean(dose_metrics[d]["psnr"]) if dose_metrics[d]["psnr"] else float('nan')
        ssim = np.mean(dose_metrics[d]["ssim"]) if dose_metrics[d]["ssim"] else float('nan')
        rmse = np.mean(dose_metrics[d]["rmse"]) if dose_metrics[d]["rmse"] else float('nan')
        print(f"Dose {d}%: PSNR={psnr:.3f}, SSIM={ssim:.4f}, RMSE={rmse:.5f}")
    return dose_metrics

# ============================================================
# ------------------------- Main -----------------------------
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mayo_root", default="/DATA/CT/LDCT_pairs/Mayo_pairs/1mm_B30")
    ap.add_argument("--use_mayo_only", action="store_true", help="If set, train only on --mayo_root")
    ap.add_argument("--out", default="runs/final")
    ap.add_argument("--epochs_teacher", type=int, default=100)
    ap.add_argument("--epochs_student", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha_ssim", type=float, default=0.2)
    ap.add_argument("--lam_lat", type=float, default=1.0)
    ap.add_argument("--lam_rec", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose dataset source
    if args.use_mayo_only:
        print(f"[DATA] Using Mayo only from: {args.mayo_root}")
        samples_all = find_mayo_samples(args.mayo_root)

    if not samples_all: raise SystemExit("No data found!")
    print(f"[Found] {len(samples_all)} samples total")

    # Train/Test Split (90/10)
    random.shuffle(samples_all)
    split = int(0.9 * len(samples_all))
    train_samples = samples_all[:split]
    test_samples = samples_all[split:]
    print(f"Training: {len(train_samples)} | Testing: {len(test_samples)}")

    # Prepare datasets
    class NDOnly(Dataset):
        def __init__(self, samples): 
            self.samples=samples
            self.MIN_HU = -1000.0
            self.MAX_HU = 1000.0
        def __len__(self): return len(self.samples)
        def __getitem__(self,i):
            nd = np.load(self.samples[i]['nd']).astype(np.float32)
            
            # UPDATED: Fixed Normalization
            nd = np.clip(nd, self.MIN_HU, self.MAX_HU)
            nd = (nd - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)
            
            t = to_tensor(nd)
            # Removed resize
            return t, 0

    ds_teacher = NDOnly(train_samples)
    ds_student = PairDataset(train_samples, pick_random_dose=True)
    
    # --- CHECK NORMALIZATION ---
    print("\n[Check] Verifying data stats on first sample...")
    chk_t, _ = ds_teacher[0]
    print(f"  Shape: {chk_t.shape}")
    print(f"  Tensor Min: {chk_t.min():.4f}, Tensor Max: {chk_t.max():.4f}")
    if chk_t.min() < -1.0 or chk_t.max() > 1.0:
        print("  WARNING: Data might not be normalized to [-1, 1]. Check your .npy values.")
    else:
        print("  OK: Data normalized correctly to [-1, 1] range.")
    print("------------------------------------------\n")

    teacher_loader = DataLoader(ds_teacher, batch_size=args.batch, shuffle=True)
    student_loader = DataLoader(ds_student, batch_size=args.batch, shuffle=True)

    # Validation loaders (for per-epoch metrics)
    val_teacher_ds = NDOnly(test_samples)
    val_student_ds = PairDataset(test_samples, pick_random_dose=False)
    val_teacher_loader = DataLoader(val_teacher_ds, batch_size=8, shuffle=False)
    val_student_loader = DataLoader(val_student_ds, batch_size=8, shuffle=False)

    # Training
    print("Stage 1: Train teacher (NDCT AE)")
    Ec,D = train_teacher(teacher_loader, device, os.path.join(args.out,"teacher"),
                         epochs=args.epochs_teacher, lr=args.lr, alpha_ssim=args.alpha_ssim,
                         val_loader=val_teacher_loader)
    print("Stage 2: Train student (LDCT encoder)")
    Es = train_student(student_loader, device, os.path.join(args.out,"student"),
                       Ec,D,epochs=args.epochs_student, lr=args.lr,
                       lam_lat=args.lam_lat, lam_rec=args.lam_rec,
                       val_loader=val_student_loader, test_samples=test_samples)

    print("✅ Training complete. Running dose-wise evaluation...")
    evaluate_model_dosewise(Es, D, test_samples, device, out_dir=os.path.join(args.out,"eval"))

if __name__ == "__main__":
    main()
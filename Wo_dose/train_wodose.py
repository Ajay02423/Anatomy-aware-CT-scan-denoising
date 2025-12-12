import os, glob, random, argparse, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

# Import MS-SSIM
try:
    from pytorch_msssim import MS_SSIM
except ImportError:
    print("❌ Error: Please install: pip install pytorch-msssim")
    exit()

# ============================================================
# ------------------------- Utils ----------------------------
# ============================================================

def set_seed(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(img_np):
    x = img_np * 2.0 - 1.0
    return torch.from_numpy(x)[None, ...].float()

@torch.no_grad()
def save_sample_comparison(pred, target, input_ld, path):
    pred = (pred[0, 0].detach().cpu().float().clamp(-1, 1) + 1) / 2
    target = (target[0, 0].detach().cpu().float().clamp(-1, 1) + 1) / 2
    input_ld = (input_ld[0, 0].detach().cpu().float().clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    titles = ["Input (LDCT)", "Predicted (No Dose Emb)", "Target (NDCT)"]
    for ax, img, title in zip(axes, [input_ld, pred, target], titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def match_size(pred, target):
    if pred.shape[-1] != target.shape[-1] or pred.shape[-2] != target.shape[-2]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, target

# ============================================================
# ------------------------- Data -----------------------------
# ============================================================

DOSES_ALLOWED = [10,25,50,70]

def find_mayo_samples(root):
    roots = sorted(glob.glob(os.path.join(root, "sample_*")))
    out = []
    for r in roots:
        nd = os.path.join(r, "NDCT_hu.npy")
        if not os.path.isfile(nd): 
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
        if doses: out.append(dict(nd=nd, doses=doses))
    return out

class PairDataset(Dataset):
    def __init__(self, samples, pick_random_dose=True, fixed_dose=None, augment=False):
        self.samples = samples
        self.pick_random_dose = pick_random_dose
        self.fixed_dose = fixed_dose
        self.augment = augment
        self.MIN_HU = -1000.0; self.MAX_HU = 1000.0
    def __len__(self): return len(self.samples)
    def normalize(self, img):
        img = np.clip(img, self.MIN_HU, self.MAX_HU)
        return ((img - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)).astype(np.float32)
    def __getitem__(self, idx):
        item = self.samples[idx]
        nd = np.load(item['nd']).astype(np.float32)
        if self.fixed_dose is not None:
            choices = [p for d,p in item['doses'] if d==self.fixed_dose]
            if not choices: d,p = random.choice(item['doses'])
            else: p = choices[0]; d = self.fixed_dose
        else: d,p = random.choice(item['doses'])
        ld = np.load(p).astype(np.float32)
        
        if self.augment:
            if random.random() > 0.5:
                nd = np.flip(nd, axis=1).copy(); ld = np.flip(ld, axis=1).copy()
            if random.random() > 0.5:
                nd = np.flip(nd, axis=0).copy(); ld = np.flip(ld, axis=0).copy()
            if random.random() > 0.5:
                k = random.randint(1, 3)
                nd = np.rot90(nd, k).copy(); ld = np.rot90(ld, k).copy()

        return to_tensor(self.normalize(nd)), to_tensor(self.normalize(ld)), float(d)

# ============================================================
# --------------- NAFNET ARCHITECTURE (No Dose Emb) ----------
# ============================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y = inp + x * self.beta

        x = y
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class DeepEncoder(nn.Module):
    def __init__(self, in_ch=1, base=32, n_blocks=8):
        super().__init__()
        
        # ABLATION: Removed Dose MLP completely
        
        self.head = nn.Conv2d(in_ch, base, 3, padding=1)
        
        self.down1 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1))
        self.down3 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, stride=2, padding=1))

        # ABLATION: Both Teacher and Student now use standard NAFBlock
        # No ConditionalNAFBlock, No SFTLayer
        self.body = nn.ModuleList([NAFBlock(base*8) for _ in range(n_blocks)])
        
        self.final = nn.Conv2d(base*8, 512, 1)

    def forward(self, x, dose_val=None): 
        # Note: dose_val is accepted as arg for compatibility but ignored
        
        features = []
        x = self.head(x); features.append(x)
        x = self.down1(x); features.append(x)
        x = self.down2(x); features.append(x)
        x = self.down3(x); features.append(x)
        
        for blk in self.body:
            x = blk(x)
        
        z = self.final(x)
        return z, features

class DeepDecoder(nn.Module):
    def __init__(self, out_ch=1, base=32, n_blocks=8):
        super().__init__()
        self.initial = nn.Conv2d(512, base*8, 1)
        self.body = nn.ModuleList([NAFBlock(base*8) for _ in range(n_blocks)])
        
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base*8, base*4, 3, stride=2, padding=1, output_padding=1))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base*4, base*2, 3, stride=2, padding=1, output_padding=1))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base*2, base, 3, stride=2, padding=1, output_padding=1))
        
        self.tail = nn.Sequential(nn.Conv2d(base, out_ch, 3, padding=1), nn.Tanh())

    def forward(self, z):
        z = self.initial(z)
        for blk in self.body: z = blk(z)
        z = self.up3(z)
        z = self.up2(z)
        z = self.up1(z)
        return self.tail(z)

# ============================================================
# ---------------------- LOSS MODULES ------------------------
# ============================================================

class SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        from pytorch_msssim import SSIM
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
    def forward(self, x, y):
        return self.ssim((x+1)/2, (y+1)/2)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:36]).eval()
        for p in self.features.parameters(): p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    def forward(self, pred, target):
        pred = (pred + 1) / 2.0; target = (target + 1) / 2.0
        if pred.shape[1] == 1: pred = pred.repeat(1, 3, 1, 1); target = target.repeat(1, 3, 1, 1)
        pred = (pred - self.mean) / self.std; target = (target - self.mean) / self.std
        return F.mse_loss(self.features(pred), self.features(target))

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1,1,3,3)
        k_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1,1,3,3)
        self.register_buffer('k_x', k_x); self.register_buffer('k_y', k_y)
    def forward(self, pred, target):
        p_x = F.conv2d(pred, self.k_x, padding=1); p_y = F.conv2d(pred, self.k_y, padding=1)
        t_x = F.conv2d(target, self.k_x, padding=1); t_y = F.conv2d(target, self.k_y, padding=1)
        return F.l1_loss(torch.abs(p_x)+torch.abs(p_y), torch.abs(t_x)+torch.abs(t_y))

# ============================================================
# ------------------------- Training -------------------------
# ============================================================

@torch.no_grad()
def evaluate_student_per_dose(Es, D, test_samples, device, batch_size=8):
    Es.eval(); D.eval()
    ssim_metric = SSIM_Loss().to(device)
    per_dose = {}
    for d in DOSES_ALLOWED:
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=d)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        psnrs, ssims = [], []
        for nd, ld, dose_val in loader:
            nd, ld = nd.to(device), ld.to(device)
            # ABLATION: Dose is ignored here
            
            z_s, _ = Es(ld) # No dose_val passed
            pred = D(z_s)
            pred, nd = match_size(pred, nd)
            
            pred01 = (pred.clamp(-1,1)+1)/2; nd01 = (nd.clamp(-1,1)+1)/2
            mse = F.mse_loss(pred01, nd01).item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-12))
            psnrs.append(psnr); ssims.append(ssim_metric(pred, nd).item())
        per_dose[d] = {"psnr": np.mean(psnrs), "ssim": np.mean(ssims)}
    return per_dose

# ---------------- Teacher ----------------
def train_teacher(loader, device, out_dir, epochs=50, lr=2e-4, alpha_ssim=0.2, val_loader=None):
    # Teacher does not use is_student flag (no dose emb anyway)
    E = DeepEncoder(in_ch=1, base=32, n_blocks=8).to(device) 
    D = DeepDecoder(out_ch=1, base=32, n_blocks=8).to(device)
    opt = torch.optim.AdamW(list(E.parameters()) + list(D.parameters()), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    
    ssim = SSIM_Loss().to(device); l1 = nn.L1Loss()
    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True); step=0

    for ep in range(1, epochs + 1):
        E.train(); D.train(); total = 0.0
        for nd, _ in loader:
            nd = nd.to(device)
            z, _ = E(nd) 
            pred = D(z)
            loss = l1(pred, nd) + alpha_ssim * (1 - ssim(pred, nd))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); step += 1; writer.add_scalar("teacher/loss", loss.item(), step)
        
        scheduler.step(); curr_lr = scheduler.get_last_lr()[0]
        print(f"[Teacher] Ep {ep} Loss: {total/len(loader):.4f} | LR: {curr_lr:.2e}")
        
        with torch.no_grad():
            z_vis, _ = E(nd[:1])
            if ep%5==0 or ep==1: save_sample_comparison(D(z_vis), nd[:1], nd[:1], os.path.join(out_dir, "samples", f"teacher_ep{ep}.png"))
    
    torch.save(E.state_dict(), os.path.join(out_dir, "teacher_Ec.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "decoder_D.pt"))
    writer.close(); return E, D

# ---------------- Student ----------------
def train_student(loader, device, out_dir, Ec, D, epochs=50, lr=2e-4, args=None, test_samples=None):
    for p in Ec.parameters(): p.requires_grad=False
    for p in D.parameters(): p.requires_grad=False
    
    # ABLATION: Student initialized same as Teacher (No Dose MLP)
    Es = DeepEncoder(in_ch=1, base=32, n_blocks=8).to(device)
    opt = torch.optim.AdamW(Es.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    
    crit_l1 = nn.L1Loss(); crit_mse = nn.MSELoss()
    crit_ssim = SSIM_Loss().to(device); crit_grad = GradientLoss().to(device)
    crit_vgg = PerceptualLoss().to(device)
    crit_msssim = MS_SSIM(data_range=2.0, size_average=True, channel=1).to(device)

    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True); step=0

    # Distillation Weight
    DISTILL_WEIGHT = 1 

    for ep in range(1, epochs + 1):
        Es.train(); total=0
        pbar = tqdm(loader, desc=f"Student Ep {ep}")
        for nd, ld, dose_val in pbar:
            nd, ld = nd.to(device), ld.to(device)
            # ABLATION: Dose tensor is ignored
            
            # Teacher
            with torch.no_grad(): zt, t_feats = Ec(nd)
            
            # Student (No Dose passed)
            zs, s_feats = Es(ld)
            pred = D(zs)
            
            # Losses
            loss_latent = crit_mse(zs, zt)
            ssim_val = crit_ssim(pred, nd)
            loss_rec = crit_l1(pred, nd) + args.alpha_ssim * (1 - ssim_val)
            loss_grad = crit_grad(pred, nd)
            loss_msssim = 1 - crit_msssim(pred, nd)
            loss_vgg = crit_vgg(pred, nd)
            
            # Feature Distillation
            loss_distill = 0.0
            for sf, tf in zip(s_feats, t_feats):
                loss_distill += crit_mse(sf, tf)
            
            loss = (args.lam_lat * loss_latent) + (args.lam_rec * loss_rec) + \
                   (args.beta1 * loss_grad) + (args.beta2 * loss_msssim) + \
                   (args.gamma1 * loss_vgg) + (DISTILL_WEIGHT * loss_distill)
            
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); step += 1; writer.add_scalar("student/loss", loss.item(), step)
            pbar.set_postfix({'L': loss.item()})
        
        scheduler.step(); curr_lr = scheduler.get_last_lr()[0]
        print(f"--> Epoch {ep} Done. LR: {curr_lr:.2e}")

        with torch.no_grad():
            if ep%5==0 or ep==1: save_sample_comparison(pred[:1], nd[:1], ld[:1], os.path.join(out_dir, "samples", f"student_ep{ep}.png"))

        if test_samples and (ep%5==0 or ep==epochs):
            per_dose = evaluate_student_per_dose(Es, D, test_samples, device)
            print(f"\n[Eval Ep {ep}]")
            for d in sorted(per_dose.keys()):
                print(f"  Dose {d}% | PSNR={per_dose[d]['psnr']:.2f} | SSIM={per_dose[d]['ssim']:.4f}")

    torch.save(Es.state_dict(), os.path.join(out_dir, "student_Es.pt")); writer.close(); return Es

# ============================================================
# ------------------------- Main -----------------------------
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mayo_root", default="/DATA/CT/LDCT_pairs/Mayo_pairs/Mayo_pairs/1mm_B30")
    ap.add_argument("--use_mayo_only", action="store_true")
    ap.add_argument("--out", default="runs/ablation_no_dose")
    ap.add_argument("--epochs_teacher", type=int, default=50)
    ap.add_argument("--epochs_student", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam_lat", type=float, default=1.0)
    ap.add_argument("--lam_rec", type=float, default=1.0)
    ap.add_argument("--alpha_ssim", type=float, default=0.5)
    ap.add_argument("--beta1", type=float, default=0.1) 
    ap.add_argument("--beta2", type=float, default=0.1)
    ap.add_argument("--gamma1", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[DATA] Using: {args.mayo_root}")
    samples_all = find_mayo_samples(args.mayo_root)
    if not samples_all: raise SystemExit("No data found!")
    
    random.shuffle(samples_all)
    split = int(0.9 * len(samples_all))
    train_samples = samples_all[:split]
    test_samples = samples_all[split:]

    class NDOnly(Dataset):
        def __init__(self, samples): 
            self.samples=samples; self.MIN_HU=-1000.0; self.MAX_HU=1000.0
        def __len__(self): return len(self.samples)
        def __getitem__(self,i):
            nd = np.load(self.samples[i]['nd']).astype(np.float32)
            nd = np.clip(nd, self.MIN_HU, self.MAX_HU)
            return to_tensor((nd - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)), 0

    ds_teacher = NDOnly(train_samples)
    ds_student = PairDataset(train_samples, pick_random_dose=True, augment=True)
    
    teacher_loader = DataLoader(ds_teacher, batch_size=args.batch, shuffle=True)
    student_loader = DataLoader(ds_student, batch_size=args.batch, shuffle=True)

    print("Stage 1: Train Teacher (NAFNet)")
    Ec,D = train_teacher(teacher_loader, device, os.path.join(args.out,"teacher"),
                         epochs=args.epochs_teacher, lr=args.lr, alpha_ssim=args.alpha_ssim)
                         
    print("Stage 2: Train Student (NAFNet - NO Dose Conditioning)")
    Es = train_student(student_loader, device, os.path.join(args.out,"student"),
                       Ec,D,epochs=args.epochs_student, lr=args.lr,
                       args=args, test_samples=test_samples)
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
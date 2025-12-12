import os, glob, random, argparse, math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision import models

# Import MS-SSIM
try:
    from pytorch_msssim import MS_SSIM
except ImportError:
    print("❌ Error: Please install the library: pip install pytorch-msssim")
    exit()
class CBAM(nn.Module):
    """ Convolutional Block Attention Module """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = x * self.sigmoid(avg_out + max_out)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return out * spatial_out

class SFTLayer(nn.Module):
    """ Spatial Feature Transform for Dose Conditioning """
    def __init__(self, feature_ch, cond_ch=32):
        super().__init__()
        self.sft = nn.Sequential(
            nn.Conv2d(cond_ch, feature_ch * 2, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x, cond_map):
        scale_shift = self.sft(cond_map)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return x * (scale + 1) + shift

# --- 1. Teacher/Decoder Block (Conv + CBAM) ---
class AttentionResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels) # Added Attention to Teacher/Decoder

    def forward(self, x, dose_emb=None):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return F.relu(identity + out)

# --- 2. Student Block (Conv + SFT + CBAM) ---
class SotaResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.cbam = CBAM(channels)
        self.sft1 = SFTLayer(channels)
        self.sft2 = SFTLayer(channels)

    def forward(self, x, dose_emb):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sft1(out, dose_emb) # SFT
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sft2(out, dose_emb) # SFT
        out = self.cbam(out)           # CBAM
        return F.relu(identity + out)

class ResEncoder(nn.Module):
    def __init__(self, in_ch=1, base=64, n_res_blocks=4, is_student=False):
        super().__init__()
        self.is_student = is_student
        
        # Dose Embedding (Only for Student)
        if self.is_student:
            self.dose_mlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU()
            )

        # Layers
        self.head = nn.Sequential(nn.Conv2d(in_ch, base, 7, padding=3, padding_mode='reflect'), nn.BatchNorm2d(base), nn.ReLU())
        self.down1 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.BatchNorm2d(base*2), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.BatchNorm2d(base*4), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, stride=2, padding=1), nn.BatchNorm2d(base*8), nn.ReLU())

        # Bottleneck: SOTA for Student, Attention for Teacher
        if self.is_student:
            self.res_blocks = nn.ModuleList([SotaResBlock(base*8) for _ in range(n_res_blocks)])
        else:
            self.res_blocks = nn.ModuleList([AttentionResBlock(base*8) for _ in range(n_res_blocks)])
        
        self.final = nn.Conv2d(base*8, 512, 1)

    def forward(self, x, dose_val=None):
        # Prepare embedding if student
        dose_emb_vec = None
        if self.is_student and dose_val is not None:
            emb = self.dose_mlp(dose_val) 
            dose_emb_vec = emb.unsqueeze(-1).unsqueeze(-1) # (B, 32, 1, 1)

        features = []
        x = self.head(x)
        features.append(x)
        x = self.down1(x)
        features.append(x)
        x = self.down2(x)
        features.append(x)
        x = self.down3(x)
        features.append(x)
        
        # Bottleneck
        for blk in self.res_blocks:
            if self.is_student:
                # Expand embedding to current size
                b, _, h, w = x.shape
                curr_map = dose_emb_vec.expand(-1, -1, h, w)
                x = blk(x, curr_map)
            else:
                x = blk(x)
                
        z = self.final(x)
        return z, features

class ResDecoder(nn.Module):
    def __init__(self, out_ch=1, base=64, n_res_blocks=4):
        super().__init__()
        self.initial = nn.Conv2d(512, base*8, 1)
        # Decoder uses Attention Blocks
        self.res_blocks = nn.ModuleList([AttentionResBlock(base*8) for _ in range(n_res_blocks)])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base*8, base*4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base*4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*4, base*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base*2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*2, base, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(base, out_ch, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    def forward(self, z):
        z = self.initial(z)
        for blk in self.res_blocks: z = blk(z)
        z = self.up(z)
        return self.tail(z)


# ============================================================
# ------------------------- Utils ----------------------------
# ============================================================

def set_seed(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(img_np):
    x = img_np * 2.0 - 1.0
    return torch.from_numpy(x)[None, ...].float()

@torch.no_grad()
def save_png(x, path):
    x = x.detach().cpu()
    x = (x.clamp(-1,1) + 1)/2
    x = x.squeeze() 
    ndarr = (x.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(ndarr).save(path)

def match_size(pred, target):
    if pred.shape[-1] != target.shape[-1] or pred.shape[-2] != target.shape[-2]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, target

# ============================================================
# ---------------------- LOSS MODULES ------------------------
# ============================================================

class SSIM_Loss_Custom(nn.Module):
    def __init__(self, win=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(win).float() - (win-1)/2
        g = torch.exp(-(coords**2)/(2*sigma**2)); g /= g.sum()
        w = (g[:,None] @ g[None,:])[None,None,:,:]
        self.register_buffer('w', w)
        self.C1 = 0.01**2; self.C2 = 0.03**2
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

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x); self.register_buffer('kernel_y', kernel_y)
    def forward(self, pred, target):
        pred_dx = F.conv2d(pred, self.kernel_x, padding=1)
        pred_dy = F.conv2d(pred, self.kernel_y, padding=1)
        target_dx = F.conv2d(target, self.kernel_x, padding=1)
        target_dy = F.conv2d(target, self.kernel_y, padding=1)
        return F.l1_loss(torch.abs(pred_dx) + torch.abs(pred_dy), torch.abs(target_dx) + torch.abs(target_dy))

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
    def __init__(self, samples, pick_random_dose=True, fixed_dose=None):
        self.samples = samples
        self.pick_random_dose = pick_random_dose
        self.fixed_dose = fixed_dose
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
        return to_tensor(self.normalize(nd)), to_tensor(self.normalize(ld)), float(d)

# ============================================================
# --------------- UPGRADED SOTA ARCHITECTURE -----------------
#      (SFT + CBAM for Student | CBAM for Teacher)
# ============================================================


# ============================================================
# ------------------------- Training -------------------------
# ============================================================

@torch.no_grad()
def evaluate_student_per_dose(Es, D, test_samples, device, batch_size=8):
    Es.eval(); D.eval()
    ssim_metric = SSIM_Loss_Custom().to(device)
    per_dose = {}
    for d in DOSES_ALLOWED:
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=d)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        psnrs, ssims = [], []
        for nd, ld, dose_val in loader:
            nd, ld = nd.to(device), ld.to(device)
            # Dose tensor
            dose_tensor = dose_val.float().to(device).view(-1, 1) / 100.0
            
            # Forward
            z_s, _ = Es(ld, dose_val=dose_tensor)
            pred = D(z_s)
            
            pred, nd = match_size(pred, nd)
            pred01 = (pred.clamp(-1,1)+1)/2; nd01 = (nd.clamp(-1,1)+1)/2
            mse = F.mse_loss(pred01, nd01).item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-12))
            psnrs.append(psnr); ssims.append(ssim_metric(pred01, nd01).item())
            
        per_dose[d] = {"psnr": np.mean(psnrs), "ssim": np.mean(ssims)}
    return per_dose

# ---------------- Teacher ----------------
def train_teacher(loader, device, out_dir, epochs=50, lr=2e-4, alpha_ssim=0.2, val_loader=None):
    # Teacher: Attention ResNet
    E = ResEncoder(in_ch=1, base=32, n_res_blocks=3, is_student=False).to(device) 
    D = ResDecoder(out_ch=1, base=32, n_res_blocks=3).to(device)
    opt = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=lr)
    
    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    
    ssim = SSIM_Loss_Custom().to(device); l1 = nn.L1Loss()
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
        
        # Step Scheduler
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        print(f"[Teacher] Ep {ep} Loss: {total/len(loader):.4f} | LR: {curr_lr:.2e}")
        
        with torch.no_grad():
            z_vis, _ = E(nd[:1])
            if ep%5==0 or ep==1: save_png(D(z_vis), os.path.join(out_dir, "samples", f"teacher_ep{ep}.png"))
    
    torch.save(E.state_dict(), os.path.join(out_dir, "teacher_Ec.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "decoder_D.pt"))
    writer.close(); return E, D

# ---------------- Student ----------------
def train_student(loader, device, out_dir, Ec, D, epochs=50, lr=2e-4, args=None, test_samples=None):
    for p in Ec.parameters(): p.requires_grad=False
    for p in D.parameters(): p.requires_grad=False
    
    # Student: SOTA (SFT + CBAM)
    Es = ResEncoder(in_ch=1, base=32, n_res_blocks=3, is_student=True).to(device)
    opt = torch.optim.Adam(Es.parameters(), lr=lr)
    
    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    
    crit_l1 = nn.L1Loss(); crit_mse = nn.MSELoss()
    crit_ssim = SSIM_Loss_Custom().to(device); crit_grad = GradientLoss().to(device)
    crit_vgg = PerceptualLoss().to(device)
    crit_msssim = MS_SSIM(data_range=2.0, size_average=True, channel=1).to(device)

    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True); step=0

    for ep in range(1, epochs + 1):
        Es.train(); total=0
        pbar = tqdm(loader, desc=f"Student Ep {ep}")
        for nd, ld, dose_val in pbar:
            nd, ld = nd.to(device), ld.to(device)
            
            # Prepare Dose Tensor (B, 1)
            dose_tensor = dose_val.float().to(device).view(-1, 1) / 100.0
            
            # --- Forward ---
            with torch.no_grad(): 
                zt, t_feats = Ec(nd) # Teacher
            
            # Student
            zs, s_feats = Es(ld, dose_val=dose_tensor) 
            
            pred = D(zs)
            
            # --- Losses ---
            loss_latent = crit_mse(zs, zt)
            ssim_val = crit_ssim(pred, nd)
            loss_rec = crit_l1(pred, nd) + args.alpha_ssim * (1 - ssim_val)
            
            loss_grad = crit_grad(pred, nd)
            loss_msssim = 1 - crit_msssim(pred, nd)
            loss_vgg = crit_vgg(pred, nd)
            
            # Distillation
            loss_distill = 0.0
            for sf, tf in zip(s_feats, t_feats):
                loss_distill += crit_mse(sf, tf)
            
            loss = (args.lam_lat * loss_latent) + (args.lam_rec * loss_rec) + \
                   (args.beta1 * loss_grad) + (args.beta2 * loss_msssim) + \
                   (args.gamma1 * loss_vgg) + (1 * loss_distill)
            
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); step += 1; writer.add_scalar("student/loss", loss.item(), step)
            pbar.set_postfix({'L': loss.item()})
        
        # Step Scheduler
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        print(f"--> Epoch {ep} Done. LR: {curr_lr:.2e}")

        with torch.no_grad():
            if ep%5==0 or ep==1: save_png(pred[:1], os.path.join(out_dir, "samples", f"student_ep{ep}.png"))

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
    ap.add_argument("--out", default="runs/sota_cbam_sft_final_end")
    ap.add_argument("--epochs_teacher", type=int, default=100)
    ap.add_argument("--epochs_student", type=int, default=150)
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
    ds_student = PairDataset(train_samples, pick_random_dose=True)
    
    teacher_loader = DataLoader(ds_teacher, batch_size=args.batch, shuffle=True)
    student_loader = DataLoader(ds_student, batch_size=args.batch, shuffle=True)

    print("Stage 1: Train Teacher (Attention ResNet)")
    Ec,D = train_teacher(teacher_loader, device, os.path.join(args.out,"teacher"),
                         epochs=args.epochs_teacher, lr=args.lr, alpha_ssim=args.alpha_ssim)
                         
    print("Stage 2: Train Student (SOTA: CBAM + SFT + Distillation)")
    Es = train_student(student_loader, device, os.path.join(args.out,"student"),
                       Ec,D,epochs=args.epochs_student, lr=args.lr,
                       args=args, test_samples=test_samples)
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
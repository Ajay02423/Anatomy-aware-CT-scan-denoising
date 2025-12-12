import os, glob, random, argparse, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

# ============================================================
# -------------------- ARCHITECTURE DEFINITIONS --------------
#        (UPDATED TO MATCH CHECKPOINT DIMENSIONS: 32)
# ============================================================

class CBAM(nn.Module):
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
    # FIXED: Default cond_ch set to 32 to match checkpoint
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

class AttentionResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels) 
    def forward(self, x, dose_emb=None):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return F.relu(identity + out)

class SotaResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)
        # FIXED: Hardcoded cond_ch=32 to match checkpoint
        self.sft1 = SFTLayer(channels, cond_ch=32)
        self.sft2 = SFTLayer(channels, cond_ch=32)
    def forward(self, x, dose_emb):
        identity = x
        out = self.conv1(x); out = self.bn1(out)
        out = self.sft1(out, dose_emb) 
        out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out = self.sft2(out, dose_emb) 
        out = self.cbam(out)           
        return F.relu(identity + out)

class ResEncoder(nn.Module):
    def __init__(self, in_ch=1, base=64, n_res_blocks=4, is_student=False):
        super().__init__()
        self.is_student = is_student
        
        if self.is_student:
            # FIXED: Output 32 to match checkpoint
            self.dose_mlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU()
            )

        self.head = nn.Sequential(nn.Conv2d(in_ch, base, 7, padding=3, padding_mode='reflect'), nn.BatchNorm2d(base), nn.ReLU())
        self.down1 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.BatchNorm2d(base*2), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.BatchNorm2d(base*4), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, stride=2, padding=1), nn.BatchNorm2d(base*8), nn.ReLU())

        if self.is_student:
            self.res_blocks = nn.ModuleList([SotaResBlock(base*8) for _ in range(n_res_blocks)])
        else:
            self.res_blocks = nn.ModuleList([AttentionResBlock(base*8) for _ in range(n_res_blocks)])
        
        self.final = nn.Conv2d(base*8, 512, 1)

    def forward(self, x, dose_val=None):
        dose_emb_vec = None
        if self.is_student and dose_val is not None:
            emb = self.dose_mlp(dose_val) 
            dose_emb_vec = emb.unsqueeze(-1).unsqueeze(-1) 

        features = []
        x = self.head(x); features.append(x)
        x = self.down1(x); features.append(x)
        x = self.down2(x); features.append(x)
        x = self.down3(x); features.append(x)
        
        for blk in self.res_blocks:
            if self.is_student:
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
# ------------------------- UTILS ----------------------------
# ============================================================

def set_seed(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor(img_np):
    x = img_np * 2.0 - 1.0
    return torch.from_numpy(x)[None, ...].float()

def match_size(pred, target):
    if pred.shape[-1] != target.shape[-1] or pred.shape[-2] != target.shape[-2]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, target

class SSIM(nn.Module):
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

# ============================================================
# -------------------- VISUALIZATION -------------------------
# ============================================================

@torch.no_grad()
def save_sample_comparison(pred, target, input_ld, path, metrics_str=""):
    """ High Quality Matplotlib Save with metrics in title """
    pred = (pred[0, 0].detach().cpu().float().clamp(-1, 1) + 1) / 2
    target = (target[0, 0].detach().cpu().float().clamp(-1, 1) + 1) / 2
    input_ld = (input_ld[0, 0].detach().cpu().float().clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    
    axes[0].set_title("Input (LDCT)", fontsize=14, fontweight='bold')
    axes[1].set_title(f"Predicted\n{metrics_str}", fontsize=14, fontweight='bold', color='darkblue')
    axes[2].set_title("Target (NDCT)", fontsize=14, fontweight='bold')

    images = [input_ld, pred, target]
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# ============================================================
# ------------------------- DATA -----------------------------
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
# ------------------------- MAIN -----------------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mayo_root", required=True)
    parser.add_argument("--weights_dir", required=True, help="Folder containing student_Es.pt and decoder_D.pt")
    parser.add_argument("--save_images", action="store_true", help="Save first 5 images per dose")
    args = parser.parse_args()

    set_seed(1234) # Same split as training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Data
    samples_all = find_mayo_samples(args.mayo_root)
    if not samples_all:
        print(f"Error: No samples found in {args.mayo_root}")
        return
        
    random.shuffle(samples_all)
    split = int(0.9 * len(samples_all))
    test_samples = samples_all[split:]
    print(f"Test Set: {len(test_samples)} samples")

    # 2. Load Models
    # Initialize Student with is_student=True to get Dose Embedding
    # Note: n_res_blocks=3 matches your training config
    Es = ResEncoder(in_ch=1, base=32, n_res_blocks=3, is_student=True).to(device)
    D = ResDecoder(out_ch=1, base=32, n_res_blocks=3).to(device)

    # Load weights
    s_path = os.path.join(args.weights_dir, "student_Es.pt")
    d_path = os.path.join(args.weights_dir, "decoder_D.pt")
    
    # If decoder is not in student folder, check parent/teacher folder
    if not os.path.exists(d_path):
        parent_dir = os.path.dirname(args.weights_dir.rstrip('/'))
        d_path = os.path.join(parent_dir, "decoder_D.pt") # Check run root
        if not os.path.exists(d_path):
             d_path = os.path.join(parent_dir, "teacher", "decoder_D.pt") # Check teacher folder

    if not os.path.exists(s_path): raise FileNotFoundError(f"Missing {s_path}")
    if not os.path.exists(d_path): raise FileNotFoundError(f"Missing {d_path}")

    print(f"Loading Student from: {s_path}")
    print(f"Loading Decoder from: {d_path}")

    Es.load_state_dict(torch.load(s_path, map_location=device))
    D.load_state_dict(torch.load(d_path, map_location=device))
    Es.eval(); D.eval()

    # 3. Evaluation
    ssim_metric = SSIM().to(device)
    results = {}
    
    if args.save_images:
        out_img_dir = os.path.join(args.weights_dir, "eval_images_resnet")
        os.makedirs(out_img_dir, exist_ok=True)
        print(f"Saving images to: {out_img_dir}")

    print("\nRunning ResNet (MLP 32-dim) Evaluation...")
    for dose in DOSES_ALLOWED:
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=dose)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        
        psnrs, ssims, rmses = [], [], []
        
        for i, (nd, ld, dose_val) in enumerate(tqdm(loader, desc=f"Dose {dose}%")):
            nd = nd.to(device)
            ld = ld.to(device)
            
            # Prepare Dose Tensor
            dose_tensor = dose_val.float().to(device).view(-1, 1) / 100.0
            
            with torch.no_grad():
                # Inference
                z, _ = Es(ld, dose_val=dose_tensor)
                pred = D(z)
                pred, nd = match_size(pred, nd)

            # Metrics
            pred01 = (pred.clamp(-1,1)+1)/2
            nd01 = (nd.clamp(-1,1)+1)/2
            
            mse = F.mse_loss(pred01, nd01).item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-12))
            ssim_val = ssim_metric(pred01, nd01).item()
            rmse = math.sqrt(mse)
            
            psnrs.append(psnr); ssims.append(ssim_val); rmses.append(rmse)
            
            # Save samples
            if args.save_images and i < 5:
                p = os.path.join(out_img_dir, f"dose_{dose}_sample_{i}.png")
                metric_txt = f"PSNR: {psnr:.2f} | SSIM: {ssim_val:.4f}"
                save_sample_comparison(pred, nd, ld, p, metric_txt)

        results[dose] = {
            "PSNR": np.mean(psnrs),
            "SSIM": np.mean(ssims),
            "RMSE": np.mean(rmses)
        }

    # 4. Print Results
    print("\n" + "="*55)
    print("      FINAL SOTA (ResNet MLP 32) RESULTS      ")
    print("="*55)
    print(f"{'Dose (%)':<10} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'RMSE':<10}")
    print("-" * 55)
    for dose in DOSES_ALLOWED:
        res = results[dose]
        print(f"{dose:<10} | {res['PSNR']:<10.4f} | {res['SSIM']:<10.4f} | {res['RMSE']:<10.5f}")
    print("="*55)

if __name__ == "__main__":
    main()
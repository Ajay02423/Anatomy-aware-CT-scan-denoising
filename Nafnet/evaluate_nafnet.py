import os, glob, random, argparse, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

# Import MS-SSIM / SSIM
try:
    from pytorch_msssim import SSIM
except ImportError:
    print("❌ Error: Please install: pip install pytorch-msssim")
    exit()

# ============================================================
# -------------------- ARCHITECTURE DEFINITIONS --------------
#        (MUST MATCH TRAINING CODE EXACTLY)
# ============================================================

class SFTLayer(nn.Module):
    def __init__(self, feature_ch, cond_ch=64):
        super().__init__()
        self.sft = nn.Sequential(
            nn.Conv2d(cond_ch, feature_ch * 2, 1),
            nn.SiLU(inplace=True) 
        )
    def forward(self, x, cond_map):
        scale_shift = self.sft(cond_map)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return x * (scale + 1) + shift

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

class ConditionalNAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.naf = NAFBlock(c)
        self.sft = SFTLayer(c, cond_ch=64) # Inject Dose

    def forward(self, x, dose_emb):
        x = self.sft(x, dose_emb)
        return self.naf(x)

class DeepEncoder(nn.Module):
    def __init__(self, in_ch=1, base=32, n_blocks=8, is_student=False):
        super().__init__()
        self.is_student = is_student
        
        # --- MLP for Dose Embedding ---
        if self.is_student:
            self.dose_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

        self.head = nn.Conv2d(in_ch, base, 3, padding=1)
        
        self.down1 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1))
        self.down3 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, stride=2, padding=1))

        if self.is_student:
            self.body = nn.ModuleList([ConditionalNAFBlock(base*8) for _ in range(n_blocks)])
        else:
            self.body = nn.ModuleList([NAFBlock(base*8) for _ in range(n_blocks)])
        
        self.final = nn.Conv2d(base*8, 512, 1)

    def forward(self, x, dose_val=None):
        dose_emb_vec = None
        if self.is_student and dose_val is not None:
            # MLP Forward
            emb = self.dose_mlp(dose_val) 
            dose_emb_vec = emb.unsqueeze(-1).unsqueeze(-1)

        features = []
        x = self.head(x); features.append(x)
        x = self.down1(x); features.append(x)
        x = self.down2(x); features.append(x)
        x = self.down3(x); features.append(x)
        
        for blk in self.body:
            if self.is_student:
                b, _, h, w = x.shape
                curr_map = dose_emb_vec.expand(-1, -1, h, w)
                x = blk(x, curr_map)
            else:
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

class SSIM_Loss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        from pytorch_msssim import SSIM
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
    def forward(self, x, y):
        # expect x,y in [0,1]
        return self.ssim(x, y)

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
    # NAFNet + MLP Configuration
    # Important: is_student=True enables MLP and Conditional Blocks
    Es = DeepEncoder(in_ch=1, base=32, n_blocks=8, is_student=True).to(device)
    D = DeepDecoder(out_ch=1, base=32, n_blocks=8).to(device)

    # Load weights
    s_path = os.path.join(args.weights_dir, "student_Es.pt")
    d_path = os.path.join(args.weights_dir, "decoder_D.pt")
    
    # Logic to find decoder if not in same folder
    if not os.path.exists(d_path):
        parent_dir = os.path.dirname(args.weights_dir.rstrip('/'))
        d_path = os.path.join(parent_dir, "decoder_D.pt")
        if not os.path.exists(d_path):
             d_path = os.path.join(parent_dir, "teacher", "decoder_D.pt")

    if not os.path.exists(s_path): raise FileNotFoundError(f"Missing {s_path}")
    if not os.path.exists(d_path): raise FileNotFoundError(f"Missing {d_path}")

    print(f"Loading Student from: {s_path}")
    print(f"Loading Decoder from: {d_path}")

    Es.load_state_dict(torch.load(s_path, map_location=device))
    D.load_state_dict(torch.load(d_path, map_location=device))
    Es.eval(); D.eval()

    # 3. Evaluation
    ssim_metric = SSIM_Loss().to(device)
    results = {}
    
    if args.save_images:
        out_img_dir = os.path.join(args.weights_dir, "eval_images_nafnet_mlp")
        os.makedirs(out_img_dir, exist_ok=True)
        print(f"Saving images to: {out_img_dir}")

    print("\nRunning NAFNet (MLP) Evaluation...")
    for dose in DOSES_ALLOWED:
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=dose)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        
        psnrs, ssims, rmses = [], [], []
        
        for i, (nd, ld, dose_val) in enumerate(tqdm(loader, desc=f"Dose {dose}%")):
            nd = nd.to(device)
            ld = ld.to(device)
            
            # Prepare Dose Tensor (Same logic as training)
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
    print("      NAFNet (MLP) MODEL RESULTS      ")
    print("="*55)
    print(f"{'Dose (%)':<10} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'RMSE':<10}")
    print("-" * 55)
    for dose in DOSES_ALLOWED:
        res = results[dose]
        print(f"{dose:<10} | {res['PSNR']:<10.4f} | {res['SSIM']:<10.4f} | {res['RMSE']:<10.5f}")
    print("="*55)

if __name__ == "__main__":
    main()
import os, glob, random, argparse, math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

# ============================================================
# ------------------------- Utils ----------------------------
# ============================================================

def set_seed(s=1234):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def to_tensor(img_np):
    # Expects input in [0, 1], scales to [-1, 1] for Tanh
    x = img_np * 2.0 - 1.0
    return torch.from_numpy(x)[None, ...].float()

def match_size(pred, target):
    if pred.shape[-1] != target.shape[-1] or pred.shape[-2] != target.shape[-2]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, target

@torch.no_grad()
def save_sample_comparison(pred, target, input_ld, path):
    # Convert all to [0, 1] for saving
    pred = (pred.clamp(-1,1) + 1)/2
    target = (target.clamp(-1,1) + 1)/2
    input_ld = (input_ld.clamp(-1,1) + 1)/2
    
    # Grid: [Input (Noisy), Prediction (Denoised), Target (Clean)]
    grid = make_grid([input_ld[0], pred[0], target[0]], nrow=3, padding=2)
    ndarr = (grid.cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
    Image.fromarray(ndarr).save(path)

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
        if doses:
            out.append(dict(nd=nd, doses=doses))
    return out

class PairDataset(Dataset):
    def __init__(self, samples, pick_random_dose=True, fixed_dose=None):
        self.samples = samples
        self.pick_random_dose = pick_random_dose
        self.fixed_dose = fixed_dose
        self.MIN_HU = -1000.0
        self.MAX_HU = 1000.0

    def __len__(self): return len(self.samples)

    def normalize(self, img):
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

        ndw = self.normalize(nd)
        ldw = self.normalize(ld)

        return to_tensor(ndw), to_tensor(ldw), d

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
# ---------------------- Main Eval ---------------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mayo_root", required=True, help="Path to Mayo dataset (e.g., /DATA/CT/LDCT_pairs/Mayo_pairs/1mm_B30)")
    parser.add_argument("--weights_dir", required=True, help="Folder containing student_Es.pt and decoder_D.pt")
    parser.add_argument("--save_images", action="store_true", help="Save comparison images for each dose")
    args = parser.parse_args()

    # 1. Setup & Device
    set_seed(1234) # IMPORTANT: Must match training seed to get correct Test Split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Find Data
    print(f"Searching for data in: {args.mayo_root}")
    samples_all = find_lodo_samples(args.mayo_root)
    if not samples_all:
        print("Error: No samples found.")
        return

    # 3. Recreate Split (90/10)
    random.shuffle(samples_all)
    split = int(0.9 * len(samples_all))
    # We only care about the test set now
    test_samples = samples_all[split:]
    print(f"Total Samples: {len(samples_all)} | Test Split: {len(test_samples)}")

    # 4. Load Model
    print("Loading models...")
    Es = Encoder().to(device)
    D = Decoder().to(device)

    student_path = os.path.join(args.weights_dir, "student_Es.pt")
    decoder_path = os.path.join(args.weights_dir, "decoder_D.pt")

    if not os.path.exists(student_path) or not os.path.exists(decoder_path):
        print(f"Error: Checkpoints not found in {args.weights_dir}")
        return

    Es.load_state_dict(torch.load(student_path, map_location=device))
    D.load_state_dict(torch.load(decoder_path, map_location=device))
    Es.eval()
    D.eval()

    # 5. Metrics Init
    ssim_metric = SSIM().to(device)
    results = {}
    
    output_img_dir = os.path.join(args.weights_dir, "eval_images")
    if args.save_images:
        os.makedirs(output_img_dir, exist_ok=True)

    # 6. Evaluation Loop per Dose
    print("\nStarting Evaluation...")
    
    for dose in DOSES_ALLOWED:
        print(f"--> Evaluating Dose {dose}% ...")
        
        # Create dataset specifically for this dose
        ds = PairDataset(test_samples, pick_random_dose=False, fixed_dose=dose)
        loader = DataLoader(ds, batch_size=1, shuffle=False) # Batch 1 for accurate stats per image
        
        psnrs, ssims, rmses = [], [], []

        for i, (nd, ld, _) in enumerate(tqdm(loader)):
            nd = nd.to(device)
            ld = ld.to(device)

            with torch.no_grad():
                # Inference
                z = Es(ld)
                pred = D(z)
                pred, nd = match_size(pred, nd)

            # Metrics calculation (on normalized [0,1] range)
            pred01 = (pred.clamp(-1,1) + 1) / 2
            nd01 = (nd.clamp(-1,1) + 1) / 2

            # MSE
            mse = F.mse_loss(pred01, nd01).item()
            
            # PSNR
            if mse == 0: psnr = 100
            else: psnr = 10 * math.log10(1.0 / mse)
            
            # SSIM
            ssim_val = ssim_metric(pred01, nd01).item()
            
            # RMSE
            rmse = math.sqrt(mse)

            psnrs.append(psnr)
            ssims.append(ssim_val)
            rmses.append(rmse)

            # Save 1st image of batch as example
            if args.save_images and i == 0:
                save_path = os.path.join(output_img_dir, f"dose_{dose}_sample.png")
                save_sample_comparison(pred, nd, ld, save_path)

        # Average stats for this dose
        results[dose] = {
            "PSNR": np.mean(psnrs),
            "SSIM": np.mean(ssims),
            "RMSE": np.mean(rmses)
        }

    # 7. Print Final Table
    print("\n" + "="*45)
    print(f"{'Dose (%)':<10} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'RMSE':<10}")
    print("-" * 45)
    
    for dose in DOSES_ALLOWED:
        res = results[dose]
        print(f"{dose:<10} | {res['PSNR']:<10.4f} | {res['SSIM']:<10.4f} | {res['RMSE']:<10.5f}")
    print("="*45)

if __name__ == "__main__":
    main()
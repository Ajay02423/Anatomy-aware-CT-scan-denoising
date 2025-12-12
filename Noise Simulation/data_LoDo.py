#!/usr/bin/env python3
"""
Re‑simulate dose levels for the LoDoPaB‑CT dataset (parallel‑beam geometry).

What it does
------------
• Reads LoDoPaB ground truth images from HDF5 (train/val/test).
• Projects them with ASTRA (parallel‑beam) to get ideal line integrals y = A(μ).
• Samples Poisson counts with incident flux I0, scaled by dose α (0<α≤1).
• Forms post‑log observations, matches LoDoPaB normalization (divide by μ_max).
• Reconstructs via FBP to PNG/NumPy and optionally writes HDF5 “observation”
  files that mirror LoDoPaB’s structure (dataset name "data", 128 samples/file).

Key LoDoPaB settings (documented):
• Parallel beam, 1000 projection angles and 513 detector bins.
• Image size 362×362 on a 26 cm × 26 cm square domain.
• Baseline incident photon count I0 = 4096 per detector bin (before attenuation).
• Noise model: Poisson only (electronic noise neglected in LoDoPaB), but you
  may add Gaussian readout noise if you want ultra‑low dose experiments.

Notes
-----
• Units: we use meters for geometry so μ_max is in m⁻¹ and line‑integrals are
  dimensionless (Beer–Lambert consistent).
• This does not try to replicate the LoDoPaB upscaling to 1000×1000 for forward
  simulation (done to avoid inverse crime). It projects directly at 362×362.
  If you want a faithful replica, swap the forward projector for ODL with an
  upsampled grid; the rest of the pipeline stays the same.
"""

import os, glob, math, json, argparse
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---- ASTRA ----
try:
    import astra
except ImportError:
    raise SystemExit("astra-toolbox not found. Install: pip install astra-toolbox")

# ---------------- constants / defaults ----------------
MU_MAX = 81.35858         # m^-1  (dival.util.constants.MU_MAX)
FOV_M  = 0.26             # 26 cm box used by LoDoPaB
IMG_N  = 362              # image size
N_ANGLES = 1000           # number of views
N_DET    = 513            # number of detector bins
I0_BASE  = 4096.0         # photons per detector bin before attenuation

# ---------------- util ----------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def window_percentile(img, lo=1, hi=99):
    lo_v = np.percentile(img, lo)
    hi_v = np.percentile(img, hi)
    if hi_v <= lo_v:
        hi_v = lo_v + 1e-6
    x = (img - lo_v) / (hi_v - lo_v)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

import imageio

def save_png(img, path, lo=1, hi=99):
    x = window_percentile(img, lo, hi)
    x_16bit = (x * 65535).astype(np.uint16)
    imageio.imwrite(path, x_16bit)


# ---------------- ASTRA helpers (parallel geometry) ----------------

def make_parallel_geom():
    angles = np.linspace(0.0, np.pi, N_ANGLES, endpoint=False).astype(np.float32)
    det_spacing = FOV_M / N_DET  # detector spans the image box width
    proj_geom = astra.create_proj_geom('parallel', float(det_spacing), int(N_DET), angles)
    half = FOV_M / 2.0
    vol_geom = astra.create_vol_geom(int(IMG_N), int(IMG_N), -half, half, -half, half)
    return proj_geom, vol_geom, angles, det_spacing

def fwd_project_mu(mu_img, proj_geom, vol_geom, force_cpu=False):
    vol_id  = astra.data2d.create('-vol',  vol_geom, mu_img.astype(np.float32))
    sino_id = astra.data2d.create('-sino', proj_geom)
    try:
        cfg = astra.astra_dict('FP_CUDA' if not force_cpu else 'FP')
    except Exception:
        cfg = astra.astra_dict('FP')
    cfg['ProjectionDataId'] = sino_id
    cfg['VolumeDataId']     = vol_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    y = astra.data2d.get(sino_id).astype(np.float32)
    astra.algorithm.delete(alg_id); astra.data2d.delete(sino_id); astra.data2d.delete(vol_id)
    return y

def recon_fbp(
    y, proj_geom, vol_geom, filter_type='hann', force_cpu=False):
    sino_id = astra.data2d.create('-sino', proj_geom, y.astype(np.float32))
    rec_id  = astra.data2d.create('-vol',  vol_geom)
    try:
        cfg = astra.astra_dict('FBP_CUDA' if not force_cpu else 'FBP')
    except Exception:
        cfg = astra.astra_dict('FBP')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['FilterType'] = filter_type
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id).astype(np.float32)
    astra.algorithm.delete(alg_id); astra.data2d.delete(sino_id); astra.data2d.delete(rec_id)
    return rec

# ---------------- noise model ----------------

def simulate_counts_from_y(y, I0, alpha, rng, sigma_e=0.0):
    I0_eff = float(I0) * float(alpha)
    lam = np.maximum(I0_eff * np.exp(-y), 1.0)
    N = rng.poisson(lam).astype(np.float32)
    if sigma_e > 0:
        N = N + rng.normal(0.0, float(sigma_e), size=N.shape).astype(np.float32)
    return np.clip(N, 1e-6, None), I0_eff

def postlog_from_counts(N, I0_eff):
    return -np.log(np.clip(N / max(I0_eff, 1e-6), 1e-6, None)).astype(np.float32)

# ---------------- HDF5 I/O ----------------

def iter_ground_truth_images(gt_dir):
    files = sorted(glob.glob(os.path.join(gt_dir, 'ground_truth_*.hdf5')))
    for fp in files:
        with h5py.File(fp, 'r') as h5:
            data = h5['data']
            for i in range(data.shape[0]):
                yield np.array(data[i], dtype=np.float32)

def write_observation_h5(out_dir, split, tag, obs_blocks, dtype='float32'):
    ensure_dir(out_dir)
    for b, arr in enumerate(obs_blocks):
        fn = f'observation_{split}_{b:03d}_{tag}.hdf5'
        with h5py.File(os.path.join(out_dir, fn), 'w') as h5:
            h5.create_dataset('data', data=arr.astype(dtype), compression='gzip')

# ---------------- main processing ----------------

def process_split(gt_dir, out_dir, doses, I0, sigma_e, seed, write_h5=False, force_cpu=False):
    proj_geom, vol_geom, angles, det_spacing = make_parallel_geom()

    block_size = 128
    blocks = {f'{int(d*100)}': [] for d in doses}
    counts = {k: [] for k in blocks.keys()}

    rng = np.random.default_rng(seed)

    for idx, x_norm in enumerate(tqdm(iter_ground_truth_images(gt_dir), desc=os.path.basename(gt_dir), ncols=80)):
        mu = (x_norm * MU_MAX).astype(np.float32)
        y = fwd_project_mu(mu, proj_geom, vol_geom, force_cpu=force_cpu)

        rec_nd = recon_fbp(y, proj_geom, vol_geom, force_cpu=force_cpu)
        case_dir = os.path.join(out_dir, f'sample_{idx:05d}')
        ensure_dir(case_dir)
        np.save(os.path.join(case_dir, 'NDCT_mu.npy'), rec_nd)
        save_png(rec_nd, os.path.join(case_dir, 'NDCT.png'))

        for a in doses:
            N, I0_eff = simulate_counts_from_y(y, I0, a, rng, sigma_e=sigma_e)
            y_noisy = postlog_from_counts(N, I0_eff)
            y_norm = (y_noisy / MU_MAX).astype(np.float32)

            rec = recon_fbp(y_noisy, proj_geom, vol_geom, force_cpu=force_cpu)
            tag = f'LDCT_{int(a*100)}'
            np.save(os.path.join(case_dir, f'{tag}_mu.npy'), rec)
            save_png(rec, os.path.join(case_dir, f'{tag}.png'))

            key = f'{int(a*100)}'
            counts[key].append(y_norm[None, ...])
            if len(counts[key]) == block_size:
                blocks[key].append(np.concatenate(counts[key], axis=0))
                counts[key] = []

    for key in counts:
        if counts[key]:
            blocks[key].append(np.concatenate(counts[key], axis=0))
            counts[key] = []

    if write_h5:
        for key, obs_blocks in blocks.items():
            write_observation_h5(os.path.join(out_dir, 'observations_h5'), os.path.basename(gt_dir).split('_')[-1], key, obs_blocks)

    meta = dict(
        geometry='parallel',
        n_angles=N_ANGLES,
        n_det=N_DET,
        fov_m=FOV_M,
        img_n=IMG_N,
        mu_max=MU_MAX,
        I0_base=I0,
        doses=doses,
        sigma_e=sigma_e,
    )
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser('Re‑simulate dose levels for LoDoPaB ground truth')
    ap.add_argument('--lodopab_root', required=True, help='folder with ground_truth_{train,validation,test}')
    ap.add_argument('--split', choices=['train','validation','test'], default='train')
    ap.add_argument('--out_root',  default='./LoDoPaB_resim')
    ap.add_argument('--doses',     type=float, nargs='+', default=[0.10,0.25,0.50,0.70])
    ap.add_argument('--I0',        type=float, default=I0_BASE, help='incident photons per bin before attenuation')
    ap.add_argument('--sigma_e',   type=float, default=0.0, help='electronic noise std‑dev in counts units (optional)')
    ap.add_argument('--seed',      type=int, default=1234)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--write_h5',  action='store_true', help='also write LoDoPaB‑style observation HDF5 files (data)')
    args = ap.parse_args()

    gt_dir  = os.path.join(args.lodopab_root, f'ground_truth_{args.split}')
    out_dir = os.path.join(args.out_root, args.split)
    ensure_dir(out_dir)

    process_split(gt_dir, out_dir, args.doses, args.I0, args.sigma_e, args.seed,
                  write_h5=args.write_h5, force_cpu=args.force_cpu)

if __name__ == '__main__':
    main()
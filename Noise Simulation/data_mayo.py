"""
AAPM LDCT realistic simulation (fan-beam, projection-domain Poisson) per slice.

Usage example:
python simulate_ldct_fanbeam.py \
  --ndct_root "/DATA/CT/AAPM/Training_Image_Data/Training_Image_Data/1mm B30/full_1mm" \
  --out_root  "/DATA/CT/LDCT_pairs/LoDo_pairs/train" \
  --doses 0.10 0.20 0.25 0.50 0.70
"""

import os, glob, math, argparse
import numpy as np
import pydicom
from PIL import Image

# ---------- ASTRA ----------
import astra

# ----------------- helpers -----------------
def load_dicom_series(folder):
    """Load a DICOM series (folder of .IMA) -> list of dicts sorted by InstanceNumber/ImagePositionPatient."""
    files = sorted(glob.glob(os.path.join(folder, "*.IMA")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        raise RuntimeError(f"No DICOMs in {folder}")

    # read all headers first for sorting
    metas = []
    for f in files:
        ds = pydicom.dcmread(f, force=True)
        z = None
        if hasattr(ds, "ImagePositionPatient"):
            z = float(ds.ImagePositionPatient[2])
        inst = int(getattr(ds, "InstanceNumber", 0))
        metas.append((f, z, inst))
    # sort by z if available, otherwise by InstanceNumber
    if all(m[1] is not None for m in metas):
        metas.sort(key=lambda x: x[1])
    else:
        metas.sort(key=lambda x: x[2])

    # now actually load pixel data
    slices = []
    for f, _, _ in metas:
        ds = pydicom.dcmread(f, force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + inter
        # spacing
        px, py = ds.PixelSpacing  # mm
        px, py = float(px), float(py)
        # optionally get thickness; not needed for 2D projection
        st = float(getattr(ds, "SliceThickness", 1.0))
        slices.append({"hu": hu, "px": px, "py": py, "thk": st, "ds": ds})
    return slices

def hu_to_mu(hu, mu_water_cm=0.19):
    """
    HU -> μ. Using mono-energetic approximation:
    μ = μ_water * (1 + HU/1000).
    Input/Output units:
      μ_water_cm in 1/cm; convert to 1/mm to match ASTRA mm geometry.
    """
    mu_water_mm = mu_water_cm / 10.0  # 1/mm
    mu = mu_water_mm * (1.0 + hu / 1000.0)
    # clamp negatives if any numerical issues
    return np.clip(mu, 0.0, None)

def mu_to_hu(mu, mu_water_cm=0.19):
    """Inverse of hu_to_mu for visualization (HU PNGs)."""
    mu_water_mm = mu_water_cm / 10.0
    hu = 1000.0 * (mu / mu_water_mm - 1.0)
    return hu

def window_hu(img_hu, center=40, width=400):
    img_hu = np.nan_to_num(img_hu, nan=0, posinf=0, neginf=0)

    # 🔹 Clip HU values to avoid very bright bone & air overflows
    img_hu = np.clip(img_hu, center - width/2, center + width/2)

    # 🔹 Normalize to [0, 1]
    img = (img_hu - (center - width/2)) / width

    # 🔹 Gamma correction for realistic contrast
    img = np.power(img, 0.75)   # makes muscle/fat visible

    return (img * 255).astype(np.uint8)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# -------------- ASTRA fan-beam ops --------------
def astra_forward_fanflat(mu_slice, px_mm, py_mm,
                          angles, Dso, Dsd, det_count, det_pitch):
    """
    Forward project a 2D μ slice with fanflat geometry on GPU (CUDA).
    Falls back to CPU if CUDA isn't available.
    Geometry in mm. μ in 1/mm.
    """
    ny, nx = mu_slice.shape  # rows, cols

    # Volume coordinates (mm), centered
    vol_geom = astra.create_vol_geom(
        nx, ny,
        -nx*px_mm/2.0, nx*px_mm/2.0,
        -ny*py_mm/2.0, ny*py_mm/2.0
    )

    # Fan-beam projection geometry (fanflat)
    proj_geom = astra.create_proj_geom(
        'fanflat',
        det_pitch, int(det_count),
        angles.astype(np.float32),
        float(Dso), float(Dsd - Dso)
    )

    vol_id = astra.data2d.create('-vol', vol_geom, mu_slice.astype(np.float32))

    try:
        # --- GPU path ---
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        sino_id = astra.data2d.create('-sino', proj_geom)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        sino = astra.data2d.get(sino_id)
        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
    except Exception:
        # --- CPU fallback ---
        sino_id = astra.data2d.create('-sino', proj_geom)
        cfg = astra.astra_dict('FP')
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        sino = astra.data2d.get(sino_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)

    astra.data2d.delete(vol_id)
    return sino


def astra_fbp_fanflat(p_noisy, nx, ny, px_mm, py_mm,
                      angles, Dso, Dsd, det_count, det_pitch, filter_type='hann'):
    """FBP reconstruction (tries CUDA, falls back to CPU)."""
    vol_geom = astra.create_vol_geom(
        nx, ny,
        -nx*px_mm/2.0, nx*px_mm/2.0,
        -ny*py_mm/2.0, ny*py_mm/2.0
    )
    proj_geom = astra.create_proj_geom(
        'fanflat',
        det_pitch, int(det_count),
        angles.astype(np.float32),
        float(Dso), float(Dsd - Dso)
    )

    rec_id  = astra.data2d.create('-vol', vol_geom, 0)
    sino_id = astra.data2d.create('-sino', proj_geom, p_noisy.astype(np.float32))

    try:
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['FilterType'] = filter_type
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
    except Exception:
        cfg = astra.astra_dict('FBP')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['FilterType'] = filter_type
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

    recon = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    return recon

def add_poisson(p_clean, dose_frac, I0_full=1e5, sigma_e=0.0):
    """
    Adds realistic Poisson and optional electronic noise to a clean sinogram.
    This version is corrected to handle photon starvation gracefully.
    """
    I0_low = I0_full * float(dose_frac)

    # Expected photon counts based on Beer-Lambert law
    # Add a small epsilon to prevent exp(-large_number) from becoming exactly zero
    mean_counts = I0_low * np.exp(-p_clean)

    # Poisson sampling to simulate quantum noise
    # mean_counts can be float, np.random.poisson handles this correctly.
    meas = np.random.poisson(mean_counts)

    # Add electronic noise (readout noise) if specified
    if sigma_e > 0:
        # Electronic noise is typically Gaussian
        meas = meas + np.random.normal(0.0, sigma_e, meas.shape)

    # THIS IS THE KEY FIX:
    # Handle photon starvation. If measured counts are less than 1 (i.e., 0 or negative
    # due to electronic noise), clip them to 1 to prevent log(0) or log(negative).
    # This is a common and effective heuristic for FBP-based reconstruction.
    meas[meas < 1] = 1

    # Back to line integrals using the inverse of Beer-Lambert law
    p_noisy = -np.log(meas / I0_low)

    # Remove any potential NaN/Inf values that might still sneak through
    p_noisy = np.nan_to_num(p_noisy, nan=0.0, posinf=0.0, neginf=0.0)

    return p_noisy

def process_subject(subject_dir, out_root, doses, geom_cfg, n_views=1000,
                    I0_full=1e5, sigma_e=0.0, win_center=40, win_width=400,
                    start_idx=1):   # ✅ added start_idx

    slices = load_dicom_series(subject_dir)
    angles = np.linspace(0, 2 * np.pi, n_views * 2, endpoint=False)

    Dso = geom_cfg["Dso_mm"]
    Dsd = geom_cfg["Dsd_mm"]
    det_pitch = geom_cfg["det_pitch_mm"]
    det_count = geom_cfg["det_count"]

    sample_idx = start_idx    # ✅ start from global index

    for s in slices:
        hu = s["hu"]
        px, py = s["px"], s["py"]
        ny, nx = hu.shape

        mu = hu_to_mu(hu)

        p = astra_forward_fanflat(mu, px, py, angles, Dso, Dsd, det_count, det_pitch)

        # Create output folder using global sample_idx
        sample_dir = os.path.join(out_root, f"sample_{sample_idx:05d}")
        ensure_dir(sample_dir)

        # Save NDCT in HU
        np.save(os.path.join(sample_dir, "NDCT_hu.npy"), hu.astype(np.float32))
        ndct_png = window_hu(hu.astype(np.float32), center=win_center, width=win_width)
        Image.fromarray(ndct_png).save(os.path.join(sample_dir, "NDCT.png"))

        # ✅ Add real LDCT_25 if available
        subject_id = os.path.basename(os.path.dirname(subject_dir))
        quarter_path = subject_dir.replace("full_1mm", "quarter_1mm")

        if os.path.exists(quarter_path):
            try:
                quarter_slices = load_dicom_series(quarter_path)
                quarter_hu = quarter_slices[sample_idx - start_idx]["hu"]
                np.save(os.path.join(sample_dir, "LDCT_25_hu.npy"), quarter_hu.astype(np.float32))
                quarter_png = window_hu(quarter_hu, center=win_center, width=win_width)
                Image.fromarray(quarter_png).save(os.path.join(sample_dir, "LDCT_25.png"))
                print(f"  ✅ Added real LDCT_25 for sample {sample_idx:05d}")
            except:
                print(f"  ⚠️ Could not load real quarter dose for sample {sample_idx}")

        # ✅ Simulate other dose levels
        for d in doses:
            if abs(d - 0.25) < 1e-3:
                continue

            p_noisy = add_poisson(p, d, I0_full=I0_full, sigma_e=sigma_e)
            recon_mu = astra_fbp_fanflat(
                p_noisy, nx, ny, px, py, angles,
                Dso, Dsd, det_count, det_pitch,
                filter_type='ram-lak'   # ✅ unchanged as you said
            )

            recon_hu = mu_to_hu(recon_mu)

            tag = int(round(d * 100))
            np.save(os.path.join(sample_dir, f"LDCT_{tag}_hu.npy"), recon_hu.astype(np.float32))
            png = window_hu(recon_hu, center=win_center, width=win_width)
            Image.fromarray(png).save(os.path.join(sample_dir, f"LDCT_{tag}.png"))

            print(f"  ✅ Saved simulated LDCT_{tag}% (HU)")

        sample_idx += 1

    return sample_idx   # ✅ Return next index to main()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndct_root", required=True,
                    help="Folder that contains subject subfolders or a single series of .IMA")
    ap.add_argument("--out_root", required=True, help="Output root (pairs dataset)")
    ap.add_argument("--doses", nargs="+", type=float, default=[0.10, 0.20, 0.25, 0.50, 0.70])
    ap.add_argument("--views", type=int, default=1000, help="Number of view angles")
    ap.add_argument("--I0", type=float, default=1e5, help="Full-dose incident photons per ray")
    ap.add_argument("--sigma_e", type=float, default=0.0, help="Electronic noise (counts)")
    # fan-beam geometry (typical clinical-like defaults; adjust if you know your scanner)
    ap.add_argument("--Dso_mm", type=float, default=595.0, help="Source-to-isocenter (mm)")
    ap.add_argument("--Dsd_mm", type=float, default=1085.0, help="Source-to-detector (mm)")
    ap.add_argument("--det_pitch_mm", type=float, default=1.0, help="Detector element pitch (mm)")
    ap.add_argument("--det_count", type=int, default=888, help="Number of detector elements")
    args = ap.parse_args()

    ensure_dir(args.out_root)

    # detect if ndct_root has subject subfolders or is a series folder itself
    subdirs = [d for d in sorted(glob.glob(os.path.join(args.ndct_root, "*"))) if os.path.isdir(d)]
    series_dirs = []
    if subdirs:
        # collect any leaf directory that contains IMA files
        for sd in subdirs:
            if glob.glob(os.path.join(sd, "*.IMA")) or glob.glob(os.path.join(sd, "*.dcm")):
                series_dirs.append(sd)
            else:
                leafs = [d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d)]
                for lf in leafs:
                    if glob.glob(os.path.join(lf, "*.IMA")) or glob.glob(os.path.join(lf, "*.dcm")):
                        series_dirs.append(lf)
    else:
        series_dirs = [args.ndct_root]

    geom_cfg = dict(Dso_mm=args.Dso_mm, Dsd_mm=args.Dsd_mm,
                    det_pitch_mm=args.det_pitch_mm, det_count=args.det_count)

    global_idx = 1   # ✅ Start global numbering

    for sd in series_dirs:
        print(f"Processing series: {sd}")
        out_series = os.path.join(args.out_root)

        # ✅ Pass and update global index
        global_idx = process_subject(
            sd, out_series, doses=args.doses, geom_cfg=geom_cfg,
            n_views=args.views, I0_full=args.I0, sigma_e=args.sigma_e,
            start_idx=global_idx
        )
if __name__ == "__main__":
    main()

# =============================================================================
# predict_direction_axis_batch.py
# =============================================================================
# Role in the overall system:
#
# This script performs batch inference using the trained PointNetReg model.
# It is executed externally by the custom Grasshopper component (built in
# Visual Studio), which provides input point clouds and reads the predicted
# results.
#
# Workflow integration:
#
#   Grasshopper component (C#)
#        ↓
#   writes temporary input file (point clouds)
#        ↓
#   calls this script via command line:
#       python predict_direction_axis_batch.py --input ... --output ... --checkpoint ...
#        ↓
#   this script:
#       - loads the trained PointNet model checkpoint (.pt)
#       - reads all beam point clouds from the input file
#       - runs inference on CPU or GPU
#       - predicts:
#           v = axis direction (unit vector)
#           c = axis point (canonical, perpendicular to v)
#       - writes results to output file
#        ↓
#   Grasshopper component reads output file
#        ↓
#   Grasshopper visualizes or uses predicted axis
#
# Notes:
# - This script does NOT train the model. It only performs inference.
# - The model checkpoint must match the PointNetReg architecture defined here.
# - Input/output files are used as a bridge between Grasshopper (C#) and Python.
# - Designed for batch processing of multiple beams for efficiency.
#
# Output fields:
#   c : predicted point on axis
#   v : predicted axis direction (normalized)
#   t : reserved (currently unused, always 0)
#
# This script is part of the runtime inference pipeline, not the training pipeline.
# =============================================================================


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== PointNetReg (v + c output) ==========================

class PointNetReg(nn.Module):
    def __init__(self, feat_dims=(64, 128, 256), head_dims=(256, 128), dropout_p=0.3):
        super().__init__()
        c1, c2, c3 = feat_dims

        self.conv1 = nn.Conv1d(3, c1, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(c1)

        self.conv2 = nn.Conv1d(c1, c2, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(c2)

        self.conv3 = nn.Conv1d(c2, c3, 1, bias=False)
        self.bn3   = nn.BatchNorm1d(c3)

        g = c3
        h1, h2 = head_dims

        self.fc1   = nn.Linear(g, h1, bias=False)
        self.bn4   = nn.BatchNorm1d(h1)
        self.drop1 = nn.Dropout(dropout_p)

        self.fc2   = nn.Linear(h1, h2, bias=False)
        self.bn5   = nn.BatchNorm1d(h2)
        self.drop2 = nn.Dropout(dropout_p)

        # Predict raw (v, c): 6 values
        self.fc_out = nn.Linear(h2, 6)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        # x: (B, N, 3)
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected input shape (B, N, 3), got {tuple(x.shape)}")

        x = x.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # (B, c3)

        x = F.relu(self.bn4(self.fc1(x))); x = self.drop1(x)
        x = F.relu(self.bn5(self.fc2(x))); x = self.drop2(x)

        raw = self.fc_out(x)    # (B, 6)
        v_raw = raw[:, 0:3]
        c_raw = raw[:, 3:6]

        v = F.normalize(v_raw, dim=1, eps=1e-8)

        # Canonicalize c so that c ⟂ v
        dot = (c_raw * v).sum(dim=1, keepdim=True)
        c = c_raw - dot * v

        return v, c


# ===================== Checkpoint loader ===================================

def load_checkpoint(checkpoint_path: str, device: torch.device) -> PointNetReg:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PointNetReg().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ===================== IO: parse beams =====================================

def read_beams_text(path: str):
    """
    Returns list of dicts:
      { "id": int, "points": np.ndarray (Ni,3) float32 }
    """
    beams = []
    cur_id = None
    cur_pts = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("beam "):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad beam header: {line}")
                cur_id = int(parts[1])
                cur_pts = []
                continue

            if line == "endbeam":
                if cur_id is None:
                    continue
                if len(cur_pts) == 0:
                    cur_id = None
                    cur_pts = []
                    continue

                pts = np.asarray(cur_pts, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] != 3:
                    raise ValueError(f"Beam {cur_id}: Expected (N,3), got {pts.shape}")

                beams.append({"id": cur_id, "points": pts})
                cur_id = None
                cur_pts = []
                continue

            # coordinate line
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Bad point line: {line}")
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            cur_pts.append([x, y, z])

    return beams


def write_results_text(path: str, results):
    """
    results: list of dicts:
      { id, c(3), v(3), t1, t2 }
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"result {r['id']}\n")
            c = r["c"]; v = r["v"]
            f.write(f"c {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write(f"t {r['t1']:.6f} {r['t2']:.6f}\n")
            f.write("endresult\n")


# ===================== Batch inference =====================================

def pad_points(beams, target_n=None):
    """
    Builds a batch array (B, N, 3) by padding/truncating.
    If target_n is None, uses max N within the batch.
    Returns: x (B,N,3) float32, mask (B,N) bool, and list of original N.
    """
    sizes = [b["points"].shape[0] for b in beams]
    if target_n is None:
        target_n = max(sizes)

    B = len(beams)
    x = np.zeros((B, target_n, 3), dtype=np.float32)
    mask = np.zeros((B, target_n), dtype=np.bool_)

    for i, b in enumerate(beams):
        pts = b["points"]
        n = pts.shape[0]
        if n >= target_n:
            x[i, :, :] = pts[:target_n, :]
            mask[i, :] = True
        else:
            x[i, :n, :] = pts
            mask[i, :n] = True

    return x, mask, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input text file with beams/points.")
    parser.add_argument("--output", required=True, help="Path to output result text file.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    beams = read_beams_text(args.input)
    if len(beams) == 0:
        raise RuntimeError("No beams found in input file.")

    model = load_checkpoint(args.checkpoint, device)
    print(f"[INFO] loaded checkpoint: {args.checkpoint}")
    print(f"[INFO] beams = {len(beams)}")

    # Currently unused (kept for output compatibility)
    t1 = 0.0
    t2 = 0.0

    results = []
    with torch.no_grad():
        for start in range(0, len(beams), args.batch_size):
            batch = beams[start:start + args.batch_size]

            x_np, _, _ = pad_points(batch, target_n=None)   # (B,N,3)
            x = torch.from_numpy(x_np).to(device)           # float32

            v_pred, c_pred = model(x)                       # (B,3), (B,3)
            v = v_pred.detach().cpu().numpy()
            c = c_pred.detach().cpu().numpy()

            for i, b in enumerate(batch):
                results.append({
                    "id": b["id"],
                    "c": c[i],
                    "v": v[i],
                    "t1": float(t1),
                    "t2": float(t2),
                })

    write_results_text(args.output, results)
    print(f"[OK] wrote {len(results)} results to: {args.output}")


if __name__ == "__main__":
    main()

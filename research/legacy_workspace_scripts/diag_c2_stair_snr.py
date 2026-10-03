"""Diagnostic: C2 detector SNR vs stair height (GLM-5.2, 2026-08-03).

Non-evidence CPU diagnostic for direction A. Reuses the frozen C2-j3 run_cell
to collect impact-window signals at stair heights 0.03/0.05/0.07 m, with a
DIAGNOSTIC seed (99, not formal seed 3) and a single cell (cell 0). Does NOT
run the 125-candidate qualification replay and emits NO qualified/unqualified
verdict (pitfall 20: no formal-seed science adjudication preview). Only
reports per-height impact-window feature maxima vs the frozen threshold table.
"""
import os, sys, json
from pathlib import Path

REPO = Path("D:/mjlab_workspace/furp-2026-Zijie-Zhang-WheelLeggedRL/.worktrees/p2-classical-upper-bound")
ART = REPO / "docs/experiments/artifacts"
os.environ["HOPPERTREX_HYBRID_CONTROLLER_PATH"] = str(ART / "c1_schedule_candidate24_1f54968_seed1/c1_schedule.json")
os.environ["HOPPERTREX_HYBRID_CALIBRATION_PATH"] = str(ART / "hybrid_runtime_seed1/velocity_calibration_seed1.json")
os.environ["HOPPERTREX_HYBRID_POSTURE_MAP_PATH"] = str(ART / "c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json")
os.environ["HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH"] = str(ART / "c1_posture_requalification_seed1/station_calibration_seed1.json")
os.environ.pop("HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH", None)

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/hoppertrex_mjlab"))

from hoppertrex_mjlab.scripts import probe_hybrid_c2_innovation_qualification as c2j3  # noqa: E402
from hoppertrex_mjlab.scripts import probe_hybrid_c2_paired_capture_v1 as capture  # noqa: E402
from hoppertrex_mjlab.hybrid.innovation_detector import (  # noqa: E402
    parse_innovation_predictor, parse_transition_floor,
)

PRED_PATH = ART / "c2_innovation_predictor_2cccb36_seed1/c2_innovation_predictor.json"
FLOOR_PATH = ART / "c2_innovation_floor_b527766_seed2/c2_innovation_floor.json"
OUT = Path("D:/mjlab_workspace/_diag_c2_snr")
OUT.mkdir(parents=True, exist_ok=True)

predictor = parse_innovation_predictor(json.loads(PRED_PATH.read_text(encoding="utf-8-sig")))
floor = parse_transition_floor(json.loads(FLOOR_PATH.read_text(encoding="utf-8-sig")), predictor_hash=predictor.predictor_hash)
# frozen threshold table: 125 rows, idx0 = lowest factor (1.05)
table = floor["threshold_table"]
import numpy as np
import torch

pitches = [r["pitch_rate_innovation_radps"] for r in table]
wheels = [r["wheel_speed_innovation_radps"] for r in table]
decels = [r["forward_deceleration_mps2"] for r in table]
HEIGHTS = [0.03, 0.05, 0.07]
THR_IDX0 = (min(pitches), min(wheels), min(decels))  # lowest-factor thresholds

def analyze_npz(path, h):
    d = np.load(path, allow_pickle=True)
    sf, ff = d["stair_features"], d["flat_features"]
    sa, fa = d["stair_active"], d["flat_active"]
    imp = d["impact_steps"]
    sw, fw, sp, fp = [], [], [], []
    for e in range(imp.shape[0]):
        is_ = int(imp[e])
        if is_ <= 3 or is_ >= sf.shape[0] - 1:
            continue
        for t in range(is_ - 1, is_ + 4):
            if 0 <= t < sf.shape[0]:
                if sa[t, e]: sw.append(sf[t, e])
                if fa[t, e]: fw.append(ff[t, e])
        for t in range(is_ - 20, is_ - 4):
            if 0 <= t < sf.shape[0]:
                if sa[t, e]: sp.append(sf[t, e])
                if fa[t, e]: fp.append(ff[t, e])
    sw, fw = np.array(sw), np.array(fw)
    tp, tw, td = THR_IDX0
    print(f"\n[diag] === H={h}m cell0 (n_pairs={imp.shape[0]}) ===")
    print(f"  THR(lowest): pitch={tp:.4f} wheel={tw:.4f} decel={td:.4f}")
    for name, cidx, thr in [("pitch", 0, tp), ("wheel", 1, tw), ("decel", 2, td)]:
        sm = float(np.abs(sw[:, cidx]).max()) if len(sw) else float("nan")
        fm = float(np.abs(fw[:, cidx]).max()) if len(fw) else float("nan")
        spa = np.array(sp) if sp else np.empty((0, 3))
        fpa = np.array(fp) if fp else np.empty((0, 3))
        spm = float(np.abs(spa[:, cidx]).max()) if len(spa) else float("nan")
        fpm = float(np.abs(fpa[:, cidx]).max()) if len(fpa) else float("nan")
        print(f"  {name}: stair_impact_max={sm:.4f} flat_impact_max={fm:.4f} "
              f"stair_pre={spm:.4f} flat_pre={fpm:.4f} | exc={sm>thr} snr={sm/(fpm+1e-9):.2f}")
    # 2-of-3 same-tick
    twoof3 = 0
    for e in range(imp.shape[0]):
        is_ = int(imp[e])
        for t in range(is_ - 1, is_ + 4):
            if 0 <= t < sf.shape[0] and sa[t, e]:
                v = sf[t, e]
                if (abs(v[0]) > tp) + (abs(v[1]) > tw) + (abs(v[2]) > td) >= 2:
                    twoof3 += 1
    print(f"  2-of-3 same-tick triggers (impact-1..+3): {twoof3}")

cells = c2j3.qualification_cells()
cell0 = cells[0]
print(f"[diag] cell0: height_m={cell0['height_m']:.4f} pitch_rad={cell0['pitch_rad']:.4f} vx_mps={cell0['vx_mps']}")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
run_protocol = c2j3.protocol(True, "cpu")
for H in HEIGHTS:
    print(f"\n[diag] >>> collecting H={H} cell0 (seed 99) ...", flush=True)
    cfg = capture.make_causal_env_cfg((0.0, H), c2j3.QUALIFICATION_PAIRS_PER_CELL)
    cfg.seed = 99
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        c2j3._assert_runtime_stack(env)
        raw = OUT / f"cell_h{H:.2f}.npz"
        summary, _ = c2j3.run_cell(env, predictor, cell=cell0, raw_path=raw, run_protocol=run_protocol)
        print(f"[diag] H={H} done. impact_steps={summary.get('impact_steps')}", flush=True)
        analyze_npz(raw, H)
    finally:
        env.close()
print("\n[diag] ALL DONE.")

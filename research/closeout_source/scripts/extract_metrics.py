"""Recompute closeout tables from archived records; never runs simulation."""
from pathlib import Path
import json, hashlib, math, csv
ROOT = Path(__file__).resolve().parents[1]
def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
manifest=load("metadata/source_manifest.json")
for row in manifest:
    p=ROOT/row["archive"]
    assert p.stat().st_size==row["bytes"] and sha(p)==row["sha256"], row["archive"]
r=load("evidence/stage5/seed1_stage5_robust_formal.json")
a=load("evidence/stage5/seed1_stage5_robust_formal_legs_ablated.json")
assert r["evaluation_profile"]==a["evaluation_profile"]=="formal"
assert r["git_sha"]==a["git_sha"] and r["checkpoint_file_sha256"]==a["checkpoint_file_sha256"]
assert sha(ROOT/"evidence/stage5/model_99.pt")==r["checkpoint_file_sha256"]
recovery=[]
for label,record in [("Hybrid residual",r),("Legs masked",a)]:
    m=record["metrics"]["stage5_recovery_center_8x"]
    improvement=1-m["candidate_recovery_time_s"]/m["baseline_recovery_time_s"]
    check=next(c for c in record["checks"] if c["name"]=="hard_regime_fractional_improvement:recovery:recovery_time_s")
    assert math.isclose(improvement,check["value"],abs_tol=1e-12)
    assert m["baseline_kick_event_count"]==m["candidate_kick_event_count"]==128
    assert m["candidate_terminated_event_rate"]==m["baseline_terminated_event_rate"]==0
    recovery.append(dict(label=label,baseline_s=m["baseline_recovery_time_s"],candidate_s=m["candidate_recovery_time_s"],improvement_pct=100*improvement,events_per_arm=128,seed=record["seed"],git_sha=record["git_sha"],gate_pass=record["gate_pass"]))
assert r["gate_pass"] and all(c["pass"] for c in r["checks"])
r0=load("evidence/r0/roll_boundary.json")
assert r0["evidence_eligible"] and not r0["training_eligible"]
assert len(r0["trials"])==480
assert len({(t["posture_card"],t["repeat"],t["stair_height_m"],t["env_id"]) for t in r0["trials"]})==480
r0rows=[]
for h in sorted({t["stair_height_m"] for t in r0["trials"]}):
    ts=[t for t in r0["trials"] if t["stair_height_m"]==h]
    assert len(ts)==96
    for t in ts:
        assert t["wheel_residual_abs_max"]==0
        if t["success"]: assert not t["bilateral_airborne_ever"] and not t["termination"] and not t["non_wheel_contact"]
    r0rows.append(dict(height_mm=h*1000,trials=len(ts),successes=sum(t["success"] for t in ts),unsupported_trials=sum(t["bilateral_airborne_ever"] for t in ts)))
assert r0rows[0]["successes"]==96 and r0rows[1]["successes"]==18 and r0rows[1]["unsupported_trials"]==73
c1=load("evidence/c1_original/c1_affine_full_gate.json")
assert c1["flat_gate_passed"] and c1["safety_clean"] and len(c1["cells"])==15
assert all(c["terminated_events"]==0 and c["non_wheel_contact_rate"]==0 for c in c1["cells"])
assert math.isclose(max(c["velocity_error_abs"] for c in c1["cells"]),c1["worst_velocity_error"])
c2=load("evidence/c2/qualification.json")
assert c2["completed_candidate_count"]==len(c2["candidates"])==125
assert c2["qualified_candidate_count"]==0 and all(c["timely_detection_count"]==0 for c in c2["candidates"])
camp=load("evidence/camp/model_999.progress.json")
assert camp["completed_updates"]==1000 and camp["evaluations"]==20 and camp["upper_height_m"]==0.01
wbc=load("evidence/r0c/r0c_effort_wbc_sanity_finalreview_dev.json")
assert not wbc["evidence_eligible"] and not wbc["promotion_eligible"] and not wbc["acceptance"]["all_required_checks_pass"]
for rel,entry in wbc["causal_file_manifest"].items():
    p=ROOT/"code/active_worktree"/rel
    assert p.stat().st_size==entry["size_bytes"] and sha(p)==entry["sha256"], rel
assert wbc["causal_file_manifest_pre"]==wbc["causal_file_manifest_post"]
metrics=dict(recovery=recovery,r0=r0rows,c1=dict(cells=15,velocity_error=c1["worst_velocity_error"],p95_pitch=c1["p95_pitch"],p99_pitch_rate=c1["p99_pitch_rate"],git_sha=c1["git_sha"]),c2=dict(candidates=125,qualified=0,paired_trials=288,git_sha=c2["git_sha"]),camp=dict(updates=1000,evaluations=20,upper_height_m=0.01,git_sha=camp["git_sha"]),r0c=dict(mode="sanity",evidence_eligible=False,source_files_verified=16))
(ROOT/"metadata/verified_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
with (ROOT/"metadata/recovery.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(recovery[0]));writer.writeheader();writer.writerows(recovery)
with (ROOT/"metadata/roll_boundary.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(r0rows[0]));writer.writeheader();writer.writerows(r0rows)
print(json.dumps(dict(status="PASS",source_files=len(manifest),r0_trials=480,c1_cells=15,c2_candidates=125,causal_files=16,recovery=recovery),indent=2))

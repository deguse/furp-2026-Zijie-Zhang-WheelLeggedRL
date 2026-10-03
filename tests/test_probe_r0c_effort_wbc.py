import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hoppertrex_mjlab.scripts.probe_r0c_effort_wbc import (
  _ALLOWED_ACTUATOR_ARRAYS,
  _CAUSAL_FILE_PATHS,
  _OPTION_FIELDS,
  _UNCHANGED_MODEL_ARRAYS,
  _assert_causal_manifest_unchanged,
  _causal_file_manifest,
  _compare_actuator_only_contract,
  _EffortMetrics,
  _first_failure_classification,
  _model_snapshot,
  _selected_resets,
  _validate_full_run_sanity_artifact,
  parse_args,
)


def _fake_model() -> SimpleNamespace:
  model = SimpleNamespace(
    nq=13,
    nv=12,
    nu=6,
    na=0,
    nbody=8,
    njnt=7,
    ngeom=10,
    nsensor=2,
    names=b"same-names",
  )
  model.opt = SimpleNamespace(
    gravity=np.asarray((0.0, 0.0, -9.81)),
    **{name: 1 for name in _OPTION_FIELDS},
  )
  model.opt.timestep = 0.005
  for name in _UNCHANGED_MODEL_ARRAYS:
    setattr(model, name, np.zeros((2,), dtype=np.float64))
  for name in _ALLOWED_ACTUATOR_ARRAYS:
    shape = (6,)
    if name in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm"):
      shape = (6, 10)
    elif name in ("actuator_ctrlrange", "actuator_forcerange", "actuator_trnid"):
      shape = (6, 2)
    elif name == "actuator_gear":
      shape = (6, 6)
    setattr(model, name, np.zeros(shape, dtype=np.float64))
  expected = np.asarray([
    [-97.0, 97.0],
    [-97.0, 97.0],
    [-5.8, 5.8],
    [-97.0, 97.0],
    [-97.0, 97.0],
    [-5.8, 5.8],
  ])
  model.actuator_ctrlrange[:] = expected
  model.actuator_forcerange[:] = expected
  model.actuator_gainprm[:, 0] = 1.0
  return model


class EffortProbeTest(unittest.TestCase):
  def test_parser_is_cpu_only_and_refuses_overwrite(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory) / "source.json"
      source.write_text("{}", encoding="utf-8")
      output = Path(directory) / "result.json"
      args = parse_args([
        "--source-result", str(source), "--output", str(output), "--sanity",
      ])
      self.assertEqual(args.device, "cpu")
      self.assertTrue(args.sanity)
      output.write_text("{}", encoding="utf-8")
      with self.assertRaises(SystemExit):
        parse_args([
          "--source-result", str(source), "--output", str(output), "--sanity",
        ])

  def test_parser_requires_a_hashed_sanity_artifact_for_full_mode(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory) / "source.json"
      source.write_text("{}", encoding="utf-8")
      output = Path(directory) / "result.json"
      sanity = Path(directory) / "sanity.json"
      sanity.write_text("{}", encoding="utf-8")
      base = ["--source-result", str(source), "--output", str(output)]
      with self.assertRaises(SystemExit):
        parse_args(base)
      args = parse_args([
        *base,
        "--validated-sanity-artifact", str(sanity),
        "--validated-sanity-sha256", "a" * 64,
      ])
      self.assertFalse(args.sanity)
      self.assertEqual(args.validated_sanity_artifact, sanity)
      with self.assertRaises(SystemExit):
        parse_args([
          *base,
          "--sanity",
          "--validated-sanity-artifact", str(sanity),
          "--validated-sanity-sha256", "a" * 64,
        ])

  def test_sanity_selects_one_reviewed_reset_per_height(self) -> None:
    resets = {(index, 0.0 if index < 8 else 0.0025): {"id": index}
              for index in range(16)}
    count, selected, identities = _selected_resets(resets, sanity=True)
    self.assertEqual(count, 1)
    self.assertEqual(identities, [(0, 0.0), (8, 0.0025)])
    self.assertEqual(selected, [{"id": 0}, {"id": 8}])

  def test_causal_manifest_hashes_every_registered_file(self) -> None:
    manifest = _causal_file_manifest()
    self.assertEqual(tuple(manifest), _CAUSAL_FILE_PATHS)
    repository = Path(__file__).resolve().parents[1]
    for relative, record in manifest.items():
      payload = (repository / relative).read_bytes()
      self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
      self.assertEqual(record["size_bytes"], len(payload))

  def test_causal_manifest_pre_post_gate_rejects_drift(self) -> None:
    manifest = {"file": {"sha256": "a" * 64, "size_bytes": 1}}
    _assert_causal_manifest_unchanged(manifest, dict(manifest))
    with self.assertRaisesRegex(RuntimeError, "file"):
      _assert_causal_manifest_unchanged(
        manifest,
        {"file": {"sha256": "b" * 64, "size_bytes": 1}},
      )

  def test_full_gate_accepts_only_a_passing_bound_sanity_artifact(self) -> None:
    manifest = {"file": {"sha256": "a" * 64, "size_bytes": 1}}
    payload = {
      "schema_version": 2,
      "kind": "r0c_effort_wbc_dual_backend_development_probe",
      "mode": "sanity",
      "source_result_sha256": "source-sha",
      "git_sha": "git-sha",
      "acceptance": {"all_required_checks_pass": True},
      "causal_provenance_gate_pass": True,
      "causal_file_manifest": manifest,
      "causal_file_manifest_pre": manifest,
      "causal_file_manifest_post": manifest,
    }
    with tempfile.TemporaryDirectory() as directory:
      artifact = Path(directory) / "sanity.json"
      artifact.write_text(json.dumps(payload), encoding="utf-8")
      artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
      gate = _validate_full_run_sanity_artifact(
        artifact,
        expected_sha256=artifact_sha,
        source_sha256="source-sha",
        current_manifest=manifest,
        current_git_sha="git-sha",
      )
      self.assertTrue(gate["all_required_checks_pass"])
      self.assertTrue(gate["causal_manifest_verified"])

      payload["acceptance"]["all_required_checks_pass"] = False
      artifact.write_text(json.dumps(payload), encoding="utf-8")
      failed_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
      with self.assertRaisesRegex(ValueError, "passing acceptance"):
        _validate_full_run_sanity_artifact(
          artifact,
          expected_sha256=failed_sha,
          source_sha256="source-sha",
          current_manifest=manifest,
          current_git_sha="git-sha",
        )

  def test_effort_metrics_label_the_fixed_non_active_filtered_window(self) -> None:
    summary = _EffortMetrics().summary()
    self.assertEqual(
      summary["metric_aggregation_window"],
      "fixed_settle_plus_drive_all_physics_substeps_including_inactive",
    )
    self.assertEqual(summary["controller_samples"], 0)
    self.assertEqual(summary["active_controller_samples"], 0)

  def test_actuator_audit_accepts_only_actuator_law_changes(self) -> None:
    candidate = _fake_model()
    baseline_model = _fake_model()
    baseline_model.actuator_gainprm[:, 0] = 2.0
    baseline_model.actuator_biasprm[:, 1] = -1.0
    report = _compare_actuator_only_contract(
      _model_snapshot(baseline_model), candidate,
    )
    self.assertTrue(report["expected_effort_ctrl_ranges_verified"])
    self.assertIn("actuator_gainprm", report["changed_actuator_fields"])


  def test_first_failure_accepts_mjwarp_rows_without_event_trace(self) -> None:
    row = {
      "env_id": 0,
      "stair_height_m": 0.0,
      "success": False,
      "bilateral_airborne_ever": True,
      "whole_body_effort": {
        "leg_saturated_samples": 0,
        "wheel_saturated_samples": 0,
        "control_trace": [{"qp_bilateral_minimum_satisfied": True}],
      },
    }
    failure = _first_failure_classification((("mjwarp", [row]),))
    self.assertIsNotNone(failure)
    assert failure is not None
    self.assertEqual(
      failure["category"], "contact_geometry_cannot_realize_target_load"
    )
    self.assertIsNone(failure["first_support_event"])

  def test_actuator_audit_rejects_geometry_drift(self) -> None:
    candidate = _fake_model()
    baseline_model = _fake_model()
    baseline_model.actuator_gainprm[:, 0] = 2.0
    baseline = _model_snapshot(baseline_model)
    candidate.geom_size[0] = 1.0
    with self.assertRaisesRegex(RuntimeError, "geom_size"):
      _compare_actuator_only_contract(baseline, candidate)


if __name__ == "__main__":
  unittest.main()



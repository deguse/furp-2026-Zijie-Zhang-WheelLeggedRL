from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import qr_candidate_grid
from hoppertrex_mjlab.scripts import evaluate_hybrid_c1_flat_gate as gate


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeNpz:
  def __init__(self, arrays: dict[str, np.ndarray]):
    self._arrays = arrays
    self.files = list(arrays)

  def __contains__(self, name: str) -> bool:
    return name in self._arrays

  def __getitem__(self, name: str) -> np.ndarray:
    return self._arrays[name]

  def __enter__(self) -> _FakeNpz:
    return self

  def __exit__(self, *_args: object) -> None:
    return None


class _CollectionFixture:
  def __init__(self, root: Path, *, bom: bool = True):
    self.nodes_dir = root / "c1_identification_nodes_e54bd1a_seed1"
    self.nodes_dir.mkdir()
    self.zip_path = self.nodes_dir.with_name(f"{self.nodes_dir.name}.zip")
    self.zip_path.write_bytes(b"frozen-zip-fixture")
    self.arrays = {
      name: np.zeros(shape, dtype=np.float64)
      for name, shape in gate.EXPECTED_ARRAY_SHAPES.items()
    }
    records: list[dict[str, object]] = []
    for stem in gate.NODE_STEMS:
      h_index = int(stem[6])
      p_index = int(stem[9])
      height = gate.REGISTERED_HEIGHT_NODES[h_index]
      pitch = gate.REGISTERED_PITCH_NODES[p_index]
      equilibrium = pitch + 0.001
      npz = self.nodes_dir / f"{stem}.npz"
      log = self.nodes_dir / f"{stem}.log"
      metadata_path = self.nodes_dir / f"{stem}.json"
      npz.write_bytes(f"npz:{stem}".encode())
      log.write_bytes(f"log:{stem}".encode())
      metadata = {
        "schema_version": 1,
        "git_sha": gate.EXPECTED_COLLECTION_GIT_SHA,
        "device": "cuda:0",
        "seed": 1,
        "num_envs": 32,
        "steps": 2500,
        "warmup_steps": 250,
        "hold_steps": 5,
        "balance_amplitude": 0.35,
        "heldout_fraction": 0.20,
        "height_command": height,
        "pitch_command": pitch,
        "equilibrium_pitch": equilibrium,
        "state_definition_version": "hybrid_lqr_equilibrium_pitch_v2",
        "state_names": list(gate.CONTROLLER_STATE_NAMES),
        "input_name": "actual_signed_balance_wheel_velocity_target",
        "wheel_radius": gate.NOMINAL_WHEEL_RADIUS_M,
        "valid_sample_count": 80000,
        "discarded_sample_count": 0,
        "calibration_hash": gate.EXPECTED_BINDINGS[
          "velocity_calibration_hash"
        ],
        "posture_artifact_hash": gate.EXPECTED_BINDINGS[
          "posture_artifact_hash"
        ],
        "station_calibration_hash": gate.EXPECTED_BINDINGS[
          "station_calibration_hash"
        ],
        "controller": {
          "qualified": True,
          "gain_hash": gate.EXPECTED_BINDINGS["controller_gain_hash"],
        },
      }
      metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
      records.append(
        {
          "stem": stem,
          "height_m": height,
          "pitch_rad": pitch,
          "equilibrium_pitch_rad": equilibrium,
          "valid_sample_count": 80000,
          "discarded_sample_count": 0,
          "npz_sha256": _sha256(npz),
          "metadata_sha256": _sha256(metadata_path),
          "log_sha256": _sha256(log),
        }
      )
    self.protocol = {
      "schema_version": 1,
      "kind": "c1_scheduled_identification_collection",
      "git_sha": gate.EXPECTED_COLLECTION_GIT_SHA,
      "mjlab_git_sha": gate.EXPECTED_MJLAB_GIT_SHA,
      "seed": 1,
      "device": "cuda:0",
      "protocol": gate.EXPECTED_COLLECTION_PROTOCOL,
      "bindings": gate.EXPECTED_BINDINGS,
      "nodes": records,
      "evidence_eligible": True,
      "promotion_eligible": False,
      "training_eligible": False,
      "checkpoint": None,
      "yaw_calibration_hash": None,
      "next_step": "offline_node_fit_then_registered_27_candidate_flat_gate",
    }
    self.write_protocol(bom=bom)

  def write_protocol(self, *, bom: bool = False) -> None:
    encoded = json.dumps(self.protocol).encode()
    if bom:
      encoded = b"\xef\xbb\xbf" + encoded
    (self.nodes_dir / "protocol_note.json").write_bytes(encoded)
    self.write_checksums()

  def write_checksums(self) -> None:
    paths = sorted(
      path
      for path in self.nodes_dir.iterdir()
      if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (self.nodes_dir / "SHA256SUMS.txt").write_text(
      "\n".join(lines) + "\n", encoding="ascii"
    )

  @property
  def zip_sha256(self) -> str:
    return _sha256(self.zip_path)

  def load(self):
    return patch.object(
      gate.np,
      "load",
      side_effect=lambda *_args, **_kwargs: _FakeNpz(self.arrays),
    )


def _candidate_rows(*, passed_index: int | None) -> tuple[list, list]:
  evaluated = []
  detail = []
  cells = [
    {
      "terminated_events": 0.0,
      "non_wheel_contact_rate": 0.0,
    }
    for _ in range(15)
  ]
  for index, candidate in enumerate(qr_candidate_grid()):
    passed = index == passed_index
    row = {
      "index": index,
      "q_diag": candidate["q_diag"],
      "r_diag": candidate["r_diag"],
      "worst_velocity_error": 0.001 + index * 1.0e-6,
      "p95_pitch": 0.002,
      "p99_pitch_rate": 0.003,
      "wheel_target_rate": 0.004,
      "flat_gate_passed": passed,
    }
    evaluated.append(row)
    detail.append({**row, "cells": cells, "node_facts": {}, "safety_clean": True})
  return evaluated, detail


class HybridC1FlatGateInputTest(unittest.TestCase):
  def test_reads_bom_and_plain_utf8_json(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "payload.json"
      for prefix in (b"\xef\xbb\xbf", b""):
        path.write_bytes(prefix + b'{"value": 7}')
        self.assertEqual(gate._read_json(path), {"value": 7})

  def test_verifies_zip_checksums_protocol_bindings_and_arrays(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      fixture = _CollectionFixture(Path(temp_dir))
      with fixture.load():
        protocol, nodes = gate.load_verified_nodes(
          fixture.nodes_dir,
          expected_zip_sha256=fixture.zip_sha256,
        )
      self.assertEqual(protocol["git_sha"], gate.EXPECTED_COLLECTION_GIT_SHA)
      self.assertEqual(len(nodes), 9)

  def test_rejects_zip_log_and_protocol_tampering(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      fixture = _CollectionFixture(Path(temp_dir))
      expected_zip = fixture.zip_sha256
      fixture.zip_path.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "ZIP SHA256"):
        gate.load_verified_nodes(
          fixture.nodes_dir, expected_zip_sha256=expected_zip
        )

    with tempfile.TemporaryDirectory() as temp_dir:
      fixture = _CollectionFixture(Path(temp_dir))
      (fixture.nodes_dir / "node_h0_p0.log").write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "SHA256SUMS mismatch"):
        gate.load_verified_nodes(
          fixture.nodes_dir, expected_zip_sha256=fixture.zip_sha256
        )

    with tempfile.TemporaryDirectory() as temp_dir:
      fixture = _CollectionFixture(Path(temp_dir))
      fixture.protocol["seed"] = 2
      fixture.write_protocol()
      with fixture.load(), self.assertRaisesRegex(ValueError, "collection seed"):
        gate.load_verified_nodes(
          fixture.nodes_dir, expected_zip_sha256=fixture.zip_sha256
        )

  def test_rejects_node_hash_binding_and_array_shape_tampering(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      fixture = _CollectionFixture(Path(temp_dir))
      metadata_path = fixture.nodes_dir / "node_h0_p0.json"
      metadata = json.loads(metadata_path.read_text())
      metadata["posture_artifact_hash"] = "0" * 64
      metadata_path.write_text(json.dumps(metadata))
      fixture.protocol["nodes"][0]["metadata_sha256"] = _sha256(metadata_path)
      fixture.write_protocol()
      with fixture.load(), self.assertRaisesRegex(
        ValueError, "posture_artifact_hash"
      ):
        gate.load_verified_nodes(
          fixture.nodes_dir, expected_zip_sha256=fixture.zip_sha256
        )

    with tempfile.TemporaryDirectory() as temp_dir:
      fixture = _CollectionFixture(Path(temp_dir))
      fixture.arrays["states"] = np.zeros((10, 4))
      with fixture.load(), self.assertRaisesRegex(ValueError, "shape mismatch"):
        gate.load_verified_nodes(
          fixture.nodes_dir, expected_zip_sha256=fixture.zip_sha256
        )


class HybridC1FlatGateLogicTest(unittest.TestCase):
  def test_fits_all_27_candidates_and_243_qualified_nodes(self) -> None:
    nodes = {
      stem: {"arrays": {name: np.zeros((1, 1)) for name in gate.REQUIRED_ARRAYS}}
      for stem in gate.NODE_STEMS
    }
    design = SimpleNamespace(
      controller_type="lqr",
      fallback_reasons=(),
      controllability_rank=4,
      heldout_nrmse=SimpleNamespace(maximum=0.09961356),
      gain=np.ones((1, 4)),
    )
    with patch.object(gate, "identify_controller", return_value=design) as identify:
      fitted = gate.fit_all_candidates(nodes)
    summary = gate.fit_qualification_summary(fitted)
    self.assertEqual(identify.call_count, 243)
    self.assertEqual(summary["candidate_count"], 27)
    self.assertEqual(summary["node_fit_count"], 243)
    self.assertEqual(summary["minimum_controllability_rank"], 4)
    self.assertEqual(summary["maximum_heldout_nrmse"], 0.09961356)
    self.assertEqual(summary["fallback_count"], 0)

  def test_fit_qualification_failure_is_not_a_gate_result(self) -> None:
    nodes = {
      stem: {"arrays": {name: np.zeros((1, 1)) for name in gate.REQUIRED_ARRAYS}}
      for stem in gate.NODE_STEMS
    }
    design = SimpleNamespace(
      controller_type="pd",
      fallback_reasons=("controllability rank 3",),
      controllability_rank=3,
      heldout_nrmse=SimpleNamespace(maximum=0.10),
      gain=np.ones((1, 4)),
    )
    with patch.object(gate, "identify_controller", return_value=design):
      with self.assertRaisesRegex(ValueError, "not a qualified LQR"):
        gate.fit_all_candidates(nodes)

  def test_registered_15_cell_order_and_caps(self) -> None:
    cells = gate.evaluation_cells(0.05)
    self.assertEqual(len(cells), 15)
    self.assertEqual(
      cells[:3],
      [
        (0.2907321708, -0.032, 0.0),
        (0.2907321708, 0.0, 0.0),
        (0.2907321708, 0.032, 0.0),
      ],
    )
    self.assertEqual(
      cells[-2:],
      [(0.3276857266, 0.032, 0.05), (0.3276857266, 0.032, -0.05)],
    )
    self.assertEqual(
      gate.registered_caps(gate.REGISTERED_FLOORS), gate.REGISTERED_CAPS
    )

  def test_frozen_floor_source_reproduces_registered_values(self) -> None:
    source = (
      Path(__file__).resolve().parents[1]
      / "docs"
      / "experiments"
      / "artifacts"
      / "c1_posture_requalification_seed1"
      / "balance_compensated_seed1.json"
    )
    floors = gate.load_registered_floors(
      source,
      expected_controller_gain_hash=gate.EXPECTED_BINDINGS[
        "controller_gain_hash"
      ],
    )
    self.assertEqual(floors, gate.REGISTERED_FLOORS)

    with tempfile.TemporaryDirectory() as temp_dir:
      tampered = Path(temp_dir) / source.name
      payload = json.loads(source.read_text())
      payload["source_probe"]["num_envs"] = 15
      tampered.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "source probe"):
        gate.load_registered_floors(
          tampered,
          expected_controller_gain_hash=gate.EXPECTED_BINDINGS[
            "controller_gain_hash"
          ],
          expected_file_sha256=_sha256(tampered),
        )

  def test_safety_performance_gate_and_lexicographic_selection(self) -> None:
    base = {
      "terminated_events": 0.0,
      "non_wheel_contact_rate": 0.0,
      "velocity_error_abs": 0.001,
      "pitch_error_abs_p95": 0.002,
      "pitch_rate_abs_p99": 0.003,
      "wheel_target_rate_rms": 0.004,
    }
    verdict = gate.aggregate_candidate([base], gate.REGISTERED_CAPS)
    self.assertTrue(verdict["flat_gate_passed"])
    self.assertFalse(
      gate.aggregate_candidate(
        [{**base, "non_wheel_contact_rate": 1.0e-9}],
        gate.REGISTERED_CAPS,
      )["flat_gate_passed"]
    )
    evaluated, _detail = _candidate_rows(passed_index=3)
    evaluated[4]["flat_gate_passed"] = True
    evaluated[4]["worst_velocity_error"] = evaluated[3]["worst_velocity_error"]
    evaluated[4]["p95_pitch"] = 0.001
    self.assertEqual(gate.select_best(evaluated), 4)

  def test_all_failed_writes_adjudication_without_selection(self) -> None:
    evaluated, detail = _candidate_rows(passed_index=None)
    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir)
      result = gate.write_evaluation_outputs(
        output,
        git_sha="1" * 40,
        mjlab_git_sha=gate.EXPECTED_MJLAB_GIT_SHA,
        collection_protocol={"git_sha": gate.EXPECTED_COLLECTION_GIT_SHA},
        bindings=gate.EXPECTED_BINDINGS,
        floors=gate.REGISTERED_FLOORS,
        caps=gate.REGISTERED_CAPS,
        run_protocol={},
        fit_summary={
          "candidate_count": 27,
          "node_fit_count": 243,
          "minimum_controllability_rank": 4,
          "maximum_heldout_nrmse": 0.09961356,
          "fallback_count": 0,
        },
        evaluated=evaluated,
        detail=detail,
      )
      self.assertEqual(
        result["classification"], "NO_QR_CANDIDATE_PASSED_FLAT_GATE"
      )
      self.assertEqual(result["next_step"], "STOP")
      self.assertTrue((output / "flat_gate_evaluation_detail.json").is_file())
      self.assertTrue((output / "flat_gate_adjudication.json").is_file())
      self.assertFalse((output / "flat_gate_selection.json").exists())

  def test_selected_writes_self_checked_selection_and_sha(self) -> None:
    evaluated, detail = _candidate_rows(passed_index=5)
    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir)
      result = gate.write_evaluation_outputs(
        output,
        git_sha="1" * 40,
        mjlab_git_sha=gate.EXPECTED_MJLAB_GIT_SHA,
        collection_protocol={"git_sha": gate.EXPECTED_COLLECTION_GIT_SHA},
        bindings=gate.EXPECTED_BINDINGS,
        floors=gate.REGISTERED_FLOORS,
        caps=gate.REGISTERED_CAPS,
        run_protocol={},
        fit_summary={
          "candidate_count": 27,
          "node_fit_count": 243,
          "minimum_controllability_rank": 4,
          "maximum_heldout_nrmse": 0.09961356,
          "fallback_count": 0,
        },
        evaluated=evaluated,
        detail=detail,
      )
      selection = output / "flat_gate_selection.json"
      self.assertEqual(result["classification"], "C1_FLAT_GATE_SELECTED")
      self.assertEqual(result["selection_sha256"], _sha256(selection))
      self.assertEqual(
        result["next_step"], "DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD"
      )

  def test_output_directory_refuses_nonempty_content(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir)
      (output / "existing.json").write_text("{}")
      with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
        gate._prepare_output_dir(output)


if __name__ == "__main__":
  unittest.main()

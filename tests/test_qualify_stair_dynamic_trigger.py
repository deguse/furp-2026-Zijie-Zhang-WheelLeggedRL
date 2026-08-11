from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
from hoppertrex_mjlab.scripts.rsl_rl import qualify_stair_dynamic_trigger as qualify
from hoppertrex_mjlab.scripts.rsl_rl import stair_dynamic_search_live_adapter
from hoppertrex_mjlab.scripts.rsl_rl.search_stair_dynamic import (
  validate_trigger_qualification,
)


def _identity() -> dict[str, object]:
  common: dict[str, object] = {
    "primary_mode": "geom",
    "primary_entity": "robot",
    "secondary_mode": "body",
    "secondary_pattern": "terrain",
    "fields": ["found", "force", "normal"],
    "reduce": "none",
    "num_slots": 8,
  }
  return {
    "left": {
      **common,
      "sensor_name": qualify.LEFT_SENSOR_NAME,
      "primary_pattern": "wheel_left_collision",
      "action_binding": qualify.LEFT_SENSOR_NAME,
    },
    "right": {
      **common,
      "sensor_name": qualify.RIGHT_SENSOR_NAME,
      "primary_pattern": "wheel_right_collision",
      "action_binding": qualify.RIGHT_SENSOR_NAME,
    },
    "objects_distinct": True,
  }


def _event(side: str, env_id: int, step: int) -> dict[str, object]:
  sensor = [18.0, 19.5, 21.0]
  return {
    "env_id": env_id,
    "side": side,
    "first_qualifying_step": step,
    "loaded_rising_step": step,
    "sensor_force_window_n": sensor,
    "fsm_force_window_n": list(sensor),
    "preceding_sensor_force_n": 7.0,
    "loaded_was_false_before": True,
  }


def _riser() -> dict[str, object]:
  return {
    "height_m": 0.01,
    "num_envs": 16,
    "settle_steps": 100,
    "drive_steps": 500,
    "step_index_base": 0,
    "reset_events": 0,
    "events": [_event("left", 0, 10), _event("right", 1, 14)],
    "trace_sha256": "3" * 64,
  }


def _fp(domain: str) -> dict[str, object]:
  flat = domain == "camp_flat_rolling"
  return {
    "events": 96_000 if flat else 128,
    "sample_events": 96_000 if flat else 86_400,
    "left_false_positives": 0,
    "right_false_positives": 0,
    "left_max_streak": 2,
    "right_max_streak": 1,
    "left_peak_metric_n": 17.4,
    "right_peak_metric_n": 16.8,
    "trace_sha256": ("4" if flat else "5") * 64,
  }


class FakeBackend:
  def __init__(self) -> None:
    self.requests = []
    self.false_positive = {
      domain: _fp(domain)
      for domain in ("camp_flat_rolling", "stage5_kick")
    }
    self.riser = _riser()
    self.identity = _identity()

  def runtime_evidence(self):
    return {
      "task": DYNAMIC_STAIR_TASK_ID,
      "evaluation_seed": 1,
      "device": "cuda:0",
      "git_sha": "a" * 40,
    }

  def sensor_identity_evidence(self):
    return self.identity

  def run_single_riser(self):
    return self.riser

  def run_false_positive(self, domain, request):
    self.requests.append((domain, request))
    return self.false_positive[domain]


class QualificationContractTest(unittest.TestCase):
  def test_happy_path_uses_registered_formal_helpers_and_search_schema(self) -> None:
    backend = FakeBackend()
    document = qualify.collect_with_backend(backend)
    qualification = document["qualification"]
    self.assertEqual(
      set(qualification),
      {
        "metric",
        "threshold_n",
        "window",
        "left_sensor_identity",
        "right_sensor_identity",
        "left_live_detected",
        "right_live_detected",
        "flat_false_positives",
        "kick_false_positives",
        "evidence_sha256",
      },
    )
    self.assertEqual(validate_trigger_qualification(qualification), qualification)
    self.assertEqual(qualification["metric"], "abs(F0*nx)")
    self.assertEqual(qualification["threshold_n"], 18.0)
    self.assertEqual(qualification["window"], 3)
    self.assertTrue(qualification["left_live_detected"])
    self.assertTrue(qualification["right_live_detected"])
    self.assertEqual(
      qualification["evidence_sha256"],
      qualify.evidence_sha256(document["evidence"]),
    )
    self.assertEqual(
      [
        (
          domain,
          request.name,
          request.profile,
          request.num_envs,
          request.steps,
          request.minimum_kick_events,
        )
        for domain, request in backend.requests
      ],
      [
        ("camp_flat_rolling", "velocity_gate_passed", "formal", 16, 3000, 0),
        ("stage5_kick", "stage5_gate_passed", "formal", 32, 3000, 128),
      ],
    )

  def test_saved_file_feeds_existing_search_live_adapter_unchanged(self) -> None:
    document = qualify.collect_with_backend(FakeBackend())
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "qualification.json"
      qualify._write_document(document, output)
      file_digest = hashlib.sha256(output.read_bytes()).hexdigest()
      variable = stair_dynamic_search_live_adapter.TRIGGER_QUALIFICATION_PATH_ENV
      with patch.dict(os.environ, {variable: str(output)}):
        result = stair_dynamic_search_live_adapter._trigger_qualification(
          file_digest
        )
      self.assertEqual(result["evidence_sha256"], file_digest)
      self.assertEqual(validate_trigger_qualification(result), result)

  def test_saved_document_is_idempotently_reverified(self) -> None:
    document = qualify.collect_with_backend(FakeBackend())
    verified = qualify.verify_document(copy.deepcopy(document))
    self.assertEqual(verified, document["qualification"])
    flat = document["evidence"]["false_positive_checks"]["camp_flat_rolling"]
    self.assertEqual(flat["protocol"]["domain"], "camp_flat_rolling")
    self.assertEqual(flat["protocol"]["events"], 96_000)

  def test_digest_binds_every_raw_evidence_surface(self) -> None:
    document = qualify.collect_with_backend(FakeBackend())
    mutations = []
    changed_trace = copy.deepcopy(document)
    changed_trace["evidence"]["single_riser_1cm"]["trace_sha256"] = "9" * 64
    mutations.append(changed_trace)
    changed_runtime = copy.deepcopy(document)
    changed_runtime["evidence"]["runtime"]["device"] = "cuda:1"
    mutations.append(changed_runtime)
    changed_summary = copy.deepcopy(document)
    changed_summary["qualification"]["evidence_sha256"] = "8" * 64
    mutations.append(changed_summary)
    for candidate in mutations:
      with self.subTest(candidate=candidate), self.assertRaises(ValueError):
        qualify.verify_document(candidate)

  def test_sensor_identity_and_live_mapping_are_strict(self) -> None:
    backend = FakeBackend()
    backend.identity["left"]["primary_pattern"] = "wheel_right_collision"
    with self.assertRaisesRegex(ValueError, "sensor identity"):
      qualify.collect_with_backend(backend)
    self.assertEqual(backend.requests, [])

    backend = FakeBackend()
    backend.riser["events"][0]["fsm_force_window_n"][1] += 0.1
    with self.assertRaisesRegex(ValueError, "FSM force"):
      qualify.collect_with_backend(backend)

  def test_each_wheel_must_have_an_exact_third_sample_rising_edge(self) -> None:
    backend = FakeBackend()
    backend.riser["events"] = [backend.riser["events"][0]]
    with self.assertRaisesRegex(ValueError, "both wheel"):
      qualify.collect_with_backend(backend)

    backend = FakeBackend()
    backend.riser["events"][1]["loaded_rising_step"] += 1
    with self.assertRaisesRegex(ValueError, "exact third"):
      qualify.collect_with_backend(backend)

    backend = FakeBackend()
    backend.riser["events"][0]["sensor_force_window_n"][0] = 17.99
    with self.assertRaisesRegex(ValueError, "sub-threshold"):
      qualify.collect_with_backend(backend)

  def test_flat_and_kick_false_positives_fail_closed(self) -> None:
    for domain, side in (
      ("camp_flat_rolling", "left"),
      ("stage5_kick", "right"),
    ):
      backend = FakeBackend()
      backend.false_positive[domain][f"{side}_false_positives"] = 1
      backend.false_positive[domain][f"{side}_max_streak"] = 3
      with self.subTest(domain=domain), self.assertRaisesRegex(
        ValueError, "18 N x 3"
      ):
        qualify.collect_with_backend(backend)

  def test_hidden_trigger_and_formal_event_drift_are_rejected(self) -> None:
    backend = FakeBackend()
    backend.false_positive["stage5_kick"]["left_max_streak"] = 3
    with self.assertRaisesRegex(ValueError, "hidden"):
      qualify.collect_with_backend(backend)

    backend = FakeBackend()
    backend.false_positive["stage5_kick"]["events"] = 127
    with self.assertRaisesRegex(ValueError, "formal binding"):
      qualify.collect_with_backend(backend)

    backend = FakeBackend()
    backend.false_positive["camp_flat_rolling"]["sample_events"] -= 1
    with self.assertRaisesRegex(ValueError, "accounting"):
      qualify.collect_with_backend(backend)

  def test_nonfinite_and_extra_fields_are_rejected(self) -> None:
    backend = FakeBackend()
    backend.false_positive["stage5_kick"]["left_peak_metric_n"] = float("nan")
    with self.assertRaisesRegex(ValueError, "finite"):
      qualify.collect_with_backend(backend)

    backend = FakeBackend()
    backend.riser["unexpected"] = True
    with self.assertRaisesRegex(ValueError, "schema drifted"):
      qualify.collect_with_backend(backend)

  def test_cli_parser_is_small_and_explicit(self) -> None:
    collect = qualify.parse_args(
      ["collect", "--device", "cuda:3", "--output", "result.json"]
    )
    self.assertEqual(collect.command, "collect")
    self.assertEqual(collect.device, "cuda:3")
    verify = qualify.parse_args(["verify", "--input", "result.json"])
    self.assertEqual(verify.command, "verify")

  def test_atomic_output_refuses_overwrite_and_leaves_no_partial(self) -> None:
    document = qualify.collect_with_backend(FakeBackend())
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "qualification.json"
      qualify._write_document(document, output)
      parsed = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(parsed, document)
      with self.assertRaises(FileExistsError):
        qualify._write_document(document, output)
      self.assertEqual(
        list(Path(temporary).glob(".qualification.json.incomplete.*")), []
      )

  def test_import_does_not_load_torch_or_mjlab(self) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    project = source / "hoppertrex_mjlab"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(source), str(project)))
    code = (
      "import sys; "
      "import hoppertrex_mjlab.scripts.rsl_rl.qualify_stair_dynamic_trigger; "
      "print(int('torch' in sys.modules), int('mjlab' in sys.modules))"
    )
    completed = subprocess.run(
      [sys.executable, "-c", code],
      check=True,
      capture_output=True,
      text=True,
      env=env,
      cwd=root,
    )
    self.assertEqual(completed.stdout.strip(), "0 0")


if __name__ == "__main__":
  unittest.main()

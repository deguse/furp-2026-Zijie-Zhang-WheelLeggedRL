"""Standard-library contract tests; no GPU or physics required."""
import tempfile
import unittest
from pathlib import Path

from hoppertrex_mjlab.hybrid import recovery_pilot as p
from hoppertrex_mjlab.scripts.recovery_c1_pilot import package_run, parse_args, require_prior_review, run_name


def ident():
  return {"task": p.TASK_ID, "git_sha": "a" * 40, "mjlab_git_sha": p.MJLAB_SHA,
          "protocol_sha256": p.PROTOCOL_HASH, "artifacts": {k: v[1] for k, v in p.ARTIFACTS.items()},
          "local_check": False, "dirty": False}


def rows(split="test", scales=p.SCALES, t=0.5):
  seeds = p.TEST_SEEDS if split == "test" else p.DEV_SEEDS
  return [{"pairing_id": p.trial_id(split, scale, seed, i), "split": split,
    "scale": scale, "reset_seed": seed, "env_id": i, "sign": 1 if i < 16 else -1,
    "initial_state_sha256": p.digest([seed, i]), "success": True,
    "recovery_time_s": t, "penalized_recovery_s": t,
    "terminated": False, "non_wheel_contact": False, "automatic_reset": False, "timeout": False}
    for scale in scales for seed in seeds for i in range(32)]


class RecoveryMetricsTest(unittest.TestCase):
  def test_zero_onset_after_hold(self):
    self.assertEqual(p.trial_metrics([True] * 300)["recovery_time_s"], 0)

  def test_hold_not_24(self):
    r = p.trial_metrics([True] * 24 + [False] * 276)
    self.assertFalse(r["success"])
    self.assertEqual(r["penalized_recovery_s"], 6)

  def test_hold_onset_not_end(self):
    self.assertEqual(p.trial_metrics([False] * 50 + [True] * 250)["recovery_time_s"], 1)

  def test_last_possible_onset(self):
    self.assertEqual(p.trial_metrics([False] * 275 + [True] * 25)["recovery_time_s"], 5.5)

  def test_any_failure_latches(self):
    for key in ("terminated", "contact", "reset"):
      with self.subTest(key=key):
        r = p.trial_metrics([True] * 300, **{key: True})
        self.assertFalse(r["success"])
        self.assertIsNone(r["recovery_time_s"])
        self.assertEqual(r["penalized_recovery_s"], 6)

  def test_no_recovery(self):
    self.assertEqual(p.trial_metrics([False] * 300)["penalized_recovery_s"], 6)

  def test_trace_shape_and_type(self):
    for values in ([True] * 299, [1] * 300):
      with self.assertRaises(ValueError):
        p.trial_metrics(values)


class PairingTest(unittest.TestCase):
  def test_complete_grid(self):
    p.validate_trials(rows(), "test")
    p.validate_trials(rows("development", (4, 8)), "development", (4, 8))

  def test_missing_duplicate_unknown(self):
    for value in (rows()[:-1], rows() + [rows()[0]], rows("development")):
      with self.assertRaises(ValueError):
        p.validate_trials(value, "test")

  def test_metadata_consistency(self):
    for field, value in (("sign", -1), ("scale", 12), ("success", "true"),
                         ("initial_state_sha256", "bad"), ("penalized_recovery_s", float("nan"))):
      sample = rows()
      sample[0][field] = value
      with self.subTest(field=field), self.assertRaises(ValueError):
        p.validate_trials(sample, "test")

  def test_false_success(self):
    sample = rows()
    sample[0]["automatic_reset"] = True
    with self.assertRaises(ValueError):
      p.validate_trials(sample, "test")

  def test_paired_comparison(self):
    r = p.compare(rows(t=1), rows(t=.8))
    self.assertAlmostEqual(r["8"]["fractional_improvement"], .2)

  def test_zero_denominator_is_none(self):
    self.assertIsNone(p.compare(rows(t=0), rows(t=0))["8"]["fractional_improvement"])

  def test_unpaired_initial_state(self):
    sample = rows()
    sample[0]["initial_state_sha256"] = "f" * 64
    with self.assertRaises(ValueError):
      p.compare(rows(), sample)


class ProvenanceTest(unittest.TestCase):
  def test_identity(self):
    p.require_identity(ident(), ident())
    for key, value in (("git_sha", "b" * 40), ("protocol_sha256", "b" * 64),
                       ("artifacts", {}), ("local_check", True), ("dirty", True)):
      actual = ident()
      actual[key] = value
      with self.subTest(key=key), self.assertRaises(ValueError):
        p.require_identity(actual, ident())

  def test_final_checkpoint(self):
    r = {**ident(), "training_seed": 11, "completed_updates": 1000,
         "config_sha256": "cfg", "initial_std": list(p.ACTION_STD),
         "zero_initialized_deterministic_mean": True}
    p.validate_checkpoint_record(r, ident(), 11, "cfg")
    for key, value in (("training_seed", 12), ("completed_updates", 999),
                       ("config_sha256", "wrong"), ("initial_std", [.6] * 6),
                       ("zero_initialized_deterministic_mean", False)):
      x = {**r, key: value}
      with self.subTest(key=key), self.assertRaises(ValueError):
        p.validate_checkpoint_record(x, ident(), 11, "cfg")

  def test_seal_tampering_and_extra_file(self):
    with tempfile.TemporaryDirectory() as t:
      d = Path(t)
      p.write_json(d / "summary.json", {"ok": True})
      p.seal(d, ident())
      p.verify_sealed(d)
      with self.assertRaises(FileExistsError):
        p.seal(d, ident())
      (d / "extra").write_text("x")
      with self.assertRaises(ValueError):
        p.verify_sealed(d)
      (d / "extra").unlink()
      (d / "summary.json").write_text("{}")
      with self.assertRaises(ValueError):
        p.verify_sealed(d)

  def test_package_complete_only_and_no_overwrite(self):
    with tempfile.TemporaryDirectory() as t:
      root = Path(t)
      d = root / "validate"
      p.write_json(d / "summary.json", {"ok": True})
      p.seal(d, ident())
      z = package_run(root, d, ident())
      self.assertTrue(z.is_file())
      self.assertTrue(z.with_suffix(".zip.sha256").is_file())
      with self.assertRaises(FileExistsError):
        package_run(root, d, ident())
      with self.assertRaises(ValueError):
        package_run(root, root.parent, ident())

  def test_approval_required_and_bound(self):
    with tempfile.TemporaryDirectory() as t:
      root = Path(t)
      base = root / "baseline"
      p.write_json(base / "summary.json", {"baseline_eligible": True})
      p.seal(base, {**ident(), "device": "cuda:0"})
      with self.assertRaises(FileNotFoundError):
        p.require_approval(root, ident())
      viewer = root / "baseline_viewer"
      p.write_json(viewer / "summary.json", {"viewer_session_completed": True})
      p.seal(viewer, {**ident(), "device": "cuda:0"})
      a = {**ident(), "viewer_manifest_sha256": p.file_sha(viewer / "manifest.json"), "baseline_manifest_sha256": p.file_sha(base / "manifest.json"),
           "viewer_verdict": "PASS", "user_feedback": "User observed stable behavior",
           "review_kind": "self_review_with_user_viewer"}
      p.write_json(root / "authorizations/baseline_review.json", a)
      p.require_approval(root, ident())
      a["viewer_verdict"] = "FAIL"
      p.write_json(root / "authorizations/baseline_review.json", a)
      with self.assertRaises(ValueError):
        p.require_approval(root, ident())


class BudgetTest(unittest.TestCase):
  def test_caps_and_seed_reservation(self):
    self.assertEqual(p.budget_remaining({"entries": []}, "train", 11), 7200)
    ledger = {"entries": [{"bucket": "train", "seed": 11, "seconds": 7000}]}
    self.assertEqual(p.budget_remaining(ledger, "train", 11), 200)
    self.assertEqual(p.budget_remaining(ledger, "train", 12), 7200)
    self.assertEqual(p.budget_remaining({"entries": [{"bucket": "baseline", "seconds": 7200}]}, "baseline", None), 0)

  def test_bad_ledger(self):
    for n in (-1, float("nan"), True):
      with self.assertRaises(ValueError):
        p.budget_remaining({"entries": [{"bucket": "train", "seconds": n}]}, "train", 11)

  def test_lease_persists_failures_and_excludes_overlap(self):
    with tempfile.TemporaryDirectory() as t:
      root = Path(t)
      with self.assertRaises(RuntimeError):
        with p.BudgetLease(root, "train", 11):
          with self.assertRaises(FileExistsError):
            with p.BudgetLease(root, "train", 12):
              pass
          raise RuntimeError("worker failed")
      ledger = p.read_json(root / "gpu_budget.json")
      self.assertEqual(len(ledger["entries"]), 1)
      self.assertIsNone(ledger["active"])
      self.assertGreaterEqual(ledger["entries"][0]["seconds"], 0)

  def test_stale_run_fails_closed(self):
    with tempfile.TemporaryDirectory() as t:
      root = Path(t)
      p.write_json(root / "gpu_budget.json", {"protocol_sha256": p.PROTOCOL_HASH, "entries": [], "active": {"pid": 1}})
      with self.assertRaises(ValueError):
        with p.BudgetLease(root, "baseline", None):
          pass


class AggregateTest(unittest.TestCase):
  def samples(self):
    return [{"training_seed": seed, "comparison": p.compare(rows(t=1), rows(t=.8)), "retention_passed": True} for seed in p.TRAIN_SEEDS]

  def test_positive_screen_still_needs_viewer(self):
    result = p.aggregate_comparisons(self.samples())
    self.assertTrue(result["numerical_continuation_criteria_passed"])
    self.assertEqual(result["decision"], "REQUIRES_REVIEW_AND_USER_VIEWER")

  def test_incomplete_is_not_negative_result(self):
    with self.assertRaises(ValueError):
      p.aggregate_comparisons(self.samples()[:2])

  def test_pressure_condition_required(self):
    sample = self.samples()
    del sample[0]["comparison"]["12"]
    with self.assertRaises(ValueError):
      p.aggregate_comparisons(sample)

  def test_retention_failure_blocks(self):
    sample = self.samples()
    sample[0]["retention_passed"] = False
    self.assertFalse(p.aggregate_comparisons(sample)["numerical_continuation_criteria_passed"])


class CliTest(unittest.TestCase):
  def test_phase_constraints(self):
    common = ["--expected-git-sha", "a" * 40, "--campaign-root", "x"]
    a = parse_args([*common, "--phase", "Validate", "--device", "cpu", "--local-check"])
    self.assertTrue(a.local_check)
    for rest in (["--phase", "Train"], ["--phase", "Baseline", "--device", "cpu"],
                 ["--phase", "Train", "--seed", "1"], ["--phase", "Package"]):
      with self.assertRaises(SystemExit):
        parse_args(common + rest)
    self.assertEqual(run_name("Train", 11), "train_seed11")

  def test_later_seed_requires_review(self):
    with tempfile.TemporaryDirectory() as t:
      require_prior_review(Path(t), 11, ident())
      with self.assertRaises(FileNotFoundError):
        require_prior_review(Path(t), 12, ident())


if __name__ == "__main__":
  unittest.main()

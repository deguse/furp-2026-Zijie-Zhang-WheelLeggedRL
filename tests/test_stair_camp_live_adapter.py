"""CPU-feasible contract and mocked-rollout tests for the live adapter."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoppertrex_mjlab.hybrid.config import STAIR_CAMP_STAGE, STAIR_CAMP_TASK_ID
from hoppertrex_mjlab.scripts.rsl_rl import evaluate_stair_camp as evaluator
from hoppertrex_mjlab.scripts.rsl_rl import stair_camp_live_adapter as adapter
from hoppertrex_mjlab.tasks import hoppertrex_hybrid_task as task_config


_GIT_SHA = "a" * 40
_CONTRACT_SHA = evaluator.STAIR_CAMP_CANONICAL_CONTRACT_SHA256
_ARTIFACTS = {
    "calibration_hash": "1" * 64,
    "controller_gain_hash": "2" * 64,
    "posture_artifact_hash": "3" * 64,
    "posture_map_hash": "4" * 64,
    "station_calibration_hash": "5" * 64,
    "yaw_calibration_hash": "6" * 64,
}


def _checkpoint() -> dict[str, object]:
    return {
        "schema_version": evaluator.EVALUATOR_SCHEMA_VERSION,
        "kind": evaluator.CHECKPOINT_ENVELOPE_KIND,
        "checkpoint_file": "D:/evidence/model_1000.pt",
        "checkpoint_file_sha256": "c" * 64,
        "training": {
            "schema_version": evaluator.STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
            "task": STAIR_CAMP_TASK_ID,
            "training_seed": 1,
            "git_sha": _GIT_SHA,
            "contract_sha256": _CONTRACT_SHA,
            "artifact_bindings": dict(_ARTIFACTS),
            "action_scales": list(STAIR_CAMP_STAGE.action_scales),
            "zero_initialized_deterministic_mean": True,
            "init_std": 0.6,
            "completed_updates": 1000,
        },
    }


def _config(
    domain: str,
    *,
    profile: str = "smoke",
    ablation: str = "baseline",
) -> dict[str, object]:
    return evaluator.make_adapter_config(
        domain=domain,
        checkpoint_envelope=_checkpoint(),
        profile=profile,
        ablation=ablation,
        device="cpu",
    )


def _pretraining_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": adapter.PRETRAINING_TRIGGER_REQUEST_KIND,
        "task": STAIR_CAMP_TASK_ID,
        "evaluation_seed": 1,
        "device": "cpu",
        "git_sha": _GIT_SHA,
        "contract_sha256": _CONTRACT_SHA,
        "artifact_bindings": dict(_ARTIFACTS),
    }


def _trial(
    cell: float,
    repeat: int,
    env_id: int,
    *,
    success: bool = True,
    terminated: bool = False,
    contact: bool = False,
    false_positive: bool = False,
    triggered: bool = True,
    pre_impact: bool = False,
) -> dict[str, object]:
    return {
        "cell": cell,
        "repeat": repeat,
        "env_id": env_id,
        "success": success,
        "terminated": terminated,
        "non_wheel_contact": contact,
        "stair_mode_false_positive": false_positive,
        "triggered": triggered,
        "pre_impact_triggered": pre_impact,
    }


class FakeBackend:
    def __init__(self) -> None:
        self.scans: list[adapter.ScanRequest] = []
        self.gates: list[adapter.GateRequest] = []
        self.fp: list[tuple[str, adapter.GateRequest]] = []
        self.fail_first = False

    def metadata(self):
        return {
            "actor_observation_width": 52,
            "critic_observation_width": 55,
            "action_width": 6,
            "mocked": True,
        }

    def run_scan(self, request):
        self.scans.append(request)
        rows = []
        for cell in request.cells:
            for repeat in range(1, request.repeats + 1):
                for env_id in range(request.num_envs_per_cell):
                    failed = self.fail_first and not rows
                    rows.append(
                        _trial(
                            cell,
                            repeat,
                            env_id,
                            success=not failed,
                            terminated=failed,
                            contact=failed,
                            false_positive=failed,
                            pre_impact=failed,
                        )
                    )
        return rows

    def run_gate(self, request):
        self.gates.append(request)
        return {
            "name": request.name,
            "num_envs": request.num_envs,
            "steps": request.steps,
            "scenario_count": request.scenario_count,
            "kick_events": request.minimum_kick_events,
            "upstream_gate_passed": True,
            "terminations": 0,
            "non_wheel_contacts": 0,
            "stair_mode_false_positives": 0,
        }

    def run_trigger_false_positive(self, domain, request):
        self.fp.append((domain, request))
        return {
            "events": request.minimum_kick_events
            if domain == "stage5_kick"
            else request.num_envs * request.steps * len(request.commands),
            "stair_mode_false_positives": 0,
        }


def _pretraining_dependencies(
    *,
    runtime_contract: str = _CONTRACT_SHA,
    runtime_git: str = _GIT_SHA,
    runtime_artifacts: dict[str, str] | None = None,
):
    actor_terms = ("actor_term",)
    critic_tail = ("critic_tail",)
    artifacts = dict(_ARTIFACTS if runtime_artifacts is None else runtime_artifacts)

    def env_cfg(*, play: bool):
        return SimpleNamespace(
            stair_camp_task_id=STAIR_CAMP_TASK_ID,
            stair_camp_training_contract=not play,
            stair_camp_zero_initialize_actor_output=True,
            observations={
                "actor": SimpleNamespace(terms={actor_terms[0]: object()}),
                "critic": SimpleNamespace(
                    terms={actor_terms[0]: object(), critic_tail[0]: object()}
                ),
            },
            artifact_bindings=dict(artifacts),
        )

    registry = SimpleNamespace(
        load_env_cfg=lambda task, play: (
            env_cfg(play=play)
            if task == STAIR_CAMP_TASK_ID
            else (_ for _ in ()).throw(AssertionError("wrong task"))
        ),
        load_rl_cfg=lambda task: (
            SimpleNamespace()
            if task == STAIR_CAMP_TASK_ID
            else (_ for _ in ()).throw(AssertionError("wrong task"))
        ),
    )
    contract = SimpleNamespace(
        STAIR_CAMP_EXPECTED_ACTOR_TERMS=actor_terms,
        STAIR_CAMP_EXPECTED_CRITIC_TAIL=critic_tail,
        STAIR_CAMP_CANONICAL_CONTRACT_SHA256=_CONTRACT_SHA,
        stair_camp_contract_hash=lambda _env, _agent: runtime_contract,
        stair_camp_artifact_bindings=lambda env: dict(env.artifact_bindings),
    )
    return SimpleNamespace(
        torch=torch,
        registry_module=registry,
        contract_module=contract,
        runner_module=SimpleNamespace(repository_git_sha=lambda: runtime_git),
    )


class ConstructionAndContractTest(unittest.TestCase):
    def test_import_is_lazy_for_torch_and_mjlab(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = root / "src"
        project = source / "hoppertrex_mjlab"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(source), str(project)))
        code = (
            "import sys; "
            "import hoppertrex_mjlab.scripts.rsl_rl.stair_camp_live_adapter; "
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

    def test_stairs_and_slope_plans_use_only_stair_camp_interface(self) -> None:
        stairs = adapter.make_terrain_plan(_config("stairs"))
        slope = adapter.make_terrain_plan(_config("slope"))
        self.assertEqual(stairs.task, STAIR_CAMP_TASK_ID)
        self.assertEqual(stairs.cells, (0.01,))
        self.assertEqual(stairs.num_envs, 1)
        self.assertEqual(slope.terrain, "inclined_plane")
        self.assertEqual(slope.cells, (5.0,))
        for plan in (stairs, slope):
            self.assertEqual(
                (
                    plan.actor_observation_width,
                    plan.critic_observation_width,
                    plan.action_width,
                ),
                (52, 55, 6),
            )
            self.assertFalse(plan.pushes_enabled)

    def test_flat_is_generated_zero_height_and_pushes_only_stage5(self) -> None:
        config = _config("flat")
        for name in evaluator.GATE_NAMES:
            plan = adapter.make_terrain_plan(config, gate_name=name)
            self.assertEqual(plan.cells, (0.0,))
            self.assertEqual(plan.terrain, "flat")
            self.assertEqual(plan.pushes_enabled, name == "stage5_gate_passed")

    def test_flat_tile_holds_the_full_registered_drive_away_from_seams(
        self,
    ) -> None:
        """Deviation minute 7: seam-safety is arithmetic, so pin the arithmetic.

        Two STOPs in a row were caused by the same seam mechanism because the
        margin was asserted from the wrong episode length instead of computed
        from the session that actually runs: flat evaluation drives
        FORMAL_GATE_STEPS continuously at the registered command speed, and
        that travel must fit inside the flat tile's half-width with real
        margin from any spawn point (the spawn is the tile center, deviation
        minute 6). The camp's own 8 m stair tile deliberately fails this
        bound for the flat drive - that is the measured defect - so this test
        also proves the flat size is not silently reverted to the stair size.
        """

        drive_steps = evaluator.FORMAL_GATE_STEPS
        control_dt = 0.02
        speed = 0.07
        travel_m = drive_steps * control_dt * speed
        self.assertAlmostEqual(travel_m, 4.2)

        flat_half_width = adapter.FLAT_EVALUATION_TERRAIN_SIZE_M[0] / 2.0
        self.assertGreaterEqual(flat_half_width - travel_m, 3.0)

        stair_half_width = task_config.STAIR_CAMP_TERRAIN_SIZE_M[0] / 2.0
        self.assertLess(stair_half_width, travel_m)

    def test_policy_interface_accepts_52_55_6_and_rejects_34(self) -> None:
        observations = {
            "actor": torch.zeros(2, 52),
            "critic": torch.zeros(2, 55),
        }
        adapter.assert_policy_interface(observations, action_width=6)
        observations["actor"] = torch.zeros(2, 34)
        with self.assertRaisesRegex(RuntimeError, "not 52"):
            adapter.assert_policy_interface(observations, action_width=6)

    def test_signed_config_mutations_and_extra_fields_are_rejected(self) -> None:
        config = _config("stairs")
        mutated = copy.deepcopy(config)
        mutated["protocol"]["drive_steps"] = 6
        with self.assertRaisesRegex(ValueError, "digest"):
            adapter.validate_adapter_config(mutated)

        interface = copy.deepcopy(config)
        interface["policy_interface"]["actor_observation_width"] = 34
        interface.pop("config_sha256")
        interface["config_sha256"] = evaluator._canonical_sha256(interface)
        with self.assertRaisesRegex(ValueError, "52-D"):
            adapter.validate_adapter_config(interface)

        extra = copy.deepcopy(config)
        extra["unregistered"] = True
        extra.pop("config_sha256")
        extra["config_sha256"] = evaluator._canonical_sha256(extra)
        with self.assertRaisesRegex(ValueError, "schema drifted"):
            adapter.validate_adapter_config(extra)

    def test_formal_slope_plan_pins_three_degrees_and_48_events(self) -> None:
        config = _config("slope", profile="formal")
        plan = adapter.make_terrain_plan(config)
        request = adapter._scan_request(config)
        self.assertEqual(plan.cells, (5.0, 10.0, 15.0))
        self.assertEqual(plan.num_envs, 48)
        self.assertEqual(request.events_per_cell, 48)
        self.assertEqual(request.travel_distance_m, 0.40)
        self.assertEqual(request.stable_steps, 25)

    def test_formal_flat_gate_plans_pin_registered_envs_steps_and_kicks(self) -> None:
        config = _config("flat", profile="formal")
        gates = {request.name: request for request in adapter._gate_requests(config)}
        self.assertEqual(
            (
                gates["standing_gate_passed"].num_envs,
                gates["standing_gate_passed"].steps,
            ),
            (16, 3000),
        )
        self.assertEqual(
            (
                gates["velocity_gate_passed"].num_envs,
                gates["velocity_gate_passed"].steps,
            ),
            (16, 3000),
        )
        self.assertEqual(
            (
                gates["stage5_gate_passed"].num_envs,
                gates["stage5_gate_passed"].steps,
                gates["stage5_gate_passed"].minimum_kick_events,
            ),
            (32, 3000, 128),
        )
        self.assertTrue(
            adapter.make_terrain_plan(
                config, gate_name="stage5_gate_passed"
            ).pushes_enabled
        )

    def test_all_eval_cfgs_keep_false_marker_and_only_stage5_copies_push(self) -> None:
        @dataclass
        class FakeAssetCfg:
            name: str

        @dataclass
        class FakeMetricsTermCfg:
            func: object
            params: dict[str, object]
            reduce: str

        def push_function(*_args, **_kwargs):
            return None

        stage = SimpleNamespace(
            push_lin_vel_x=0.32,
            push_pitch_rate=0.48,
            push_interval_s=(3.0, 5.0),
        )
        task_module = SimpleNamespace(
            HYBRID_STAGES={5: stage},
            SceneEntityCfg=FakeAssetCfg,
            envs_mdp=SimpleNamespace(push_by_setting_velocity=push_function),
        )
        canonical_push = SimpleNamespace(
            func=push_function,
            params={
                "asset_cfg": FakeAssetCfg("robot"),
                "velocity_range": {
                    "x": (-0.32, 0.32),
                    "pitch": (-0.48, 0.48),
                },
            },
            mode="interval",
            interval_range_s=(3.0, 5.0),
            is_global_time=False,
            min_step_count_between_reset=0,
        )
        training_cfg = SimpleNamespace(
            stair_camp_task_id=STAIR_CAMP_TASK_ID,
            stair_camp_training_contract=True,
            events={"push_robot": canonical_push},
        )

        def make_play_cfg():
            return SimpleNamespace(
                stair_camp_task_id=STAIR_CAMP_TASK_ID,
                stair_camp_training_contract=False,
                seed=0,
                scene=SimpleNamespace(num_envs=16, terrain=None),
                events={"reset_root_to_stair_approach": SimpleNamespace(params={})},
                curriculum={"must_not_survive": object()},
                metrics={"must_not_survive": object()},
                episode_length_s=20.0,
                actions={"hybrid_wheel_leg": SimpleNamespace()},
                observations={
                    "critic": SimpleNamespace(
                        terms={"step_height": SimpleNamespace(func=None, params={})}
                    )
                },
            )

        registry = SimpleNamespace(
            load_env_cfg=lambda task, play: (
                training_cfg
                if (task == STAIR_CAMP_TASK_ID and not play)
                else make_play_cfg()
            )
        )
        deps = SimpleNamespace(
            registry_module=registry,
            task_module=task_module,
            manager_module=SimpleNamespace(MetricsTermCfg=FakeMetricsTermCfg),
            balance_task_module=SimpleNamespace(
                non_wheel_ground_contact=lambda *_args: None,
                NON_WHEEL_GROUND_SENSOR_NAME="non_wheel",
            ),
        )
        backend = adapter._MjLabBackend(_config("flat"), deps)
        registered_calls: list[bool] = []

        def registered_configs(*, play: bool):
            registered_calls.append(play)
            return make_play_cfg(), SimpleNamespace()

        specs = (
            ("stairs", (0.01,), False),
            ("flat", (0.0,), False),
            ("slope", (5.0,), False),
            ("flat", (0.0,), True),
        )
        configs = []
        with (
            patch.object(
                backend, "_registered_configs", side_effect=registered_configs
            ),
            patch.object(backend, "_terrain_cfg", return_value="generated"),
        ):
            for domain, cells, pushes in specs:
                cfg, _agent = backend._evaluation_env_cfg(
                    domain=domain,
                    cells=cells,
                    num_envs=1,
                    pushes=pushes,
                )
                configs.append(cfg)

        self.assertEqual(registered_calls, [True, True, True, True])
        for index, cfg in enumerate(configs):
            self.assertIs(cfg.stair_camp_training_contract, False)
            self.assertEqual(cfg.curriculum, {})
            self.assertEqual(set(cfg.metrics), {adapter._LIVE_EVIDENCE_TERM_NAME})
            metric = cfg.metrics[adapter._LIVE_EVIDENCE_TERM_NAME]
            self.assertIs(metric.func, adapter._LivePreResetEvidenceMetric)
            self.assertEqual("push_robot" in cfg.events, index == 3)
            # Deviation minute 6: FLAT sessions spawn at the tile center
            # (no riser exists to approach, and the stair-approach spawn
            # measurably parks the robot at the seam); stairs and slope keep
            # the registered stair-approach spawn untouched.
            reset_params = cfg.events["reset_root_to_stair_approach"].params
            domain = specs[index][0]
            if domain == "flat":
                self.assertEqual(reset_params["x_offset_from_origin_m"], 0.0)
            else:
                self.assertNotIn("x_offset_from_origin_m", reset_params)
        self.assertEqual(configs[-1].events["push_robot"], canonical_push)
        self.assertIsNot(configs[-1].events["push_robot"], canonical_push)

        drifted = copy.deepcopy(training_cfg)
        drifted.events["push_robot"].interval_range_s = (4.0, 5.0)
        with self.assertRaisesRegex(RuntimeError, "scheduling semantics"):
            adapter._validated_stage5_push_event(drifted, task_module)

    def test_flat_plan_requires_registered_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires one fixed gate"):
            adapter.make_terrain_plan(_config("flat"))
        with self.assertRaisesRegex(ValueError, "Unknown flat gate"):
            adapter.make_terrain_plan(_config("flat"), gate_name="stage5")


class AblationTest(unittest.TestCase):
    def test_leg_off_zeros_only_four_leg_heads(self) -> None:
        def policy(_observations):
            return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        descriptor = evaluator.resolve_ablation("leg-off").to_dict()
        wrapped = adapter.apply_policy_ablation(policy, descriptor)
        self.assertTrue(
            torch.equal(
                wrapped({}),
                torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]]),
            )
        )

    def test_zero_shot_scales_mutate_only_environment_leg_authority(self) -> None:
        for name, expected in (
            ("zero-shot-scale-0.035", 0.035),
            ("zero-shot-scale-0.070", 0.070),
            ("zero-shot-scale-0.100", 0.100),
        ):
            action = SimpleNamespace(action_scales=(0.0, 0.0, 0.07, 0.07, 0.07, 0.07))
            adapter.apply_environment_ablation(
                action, evaluator.resolve_ablation(name).to_dict()
            )
            self.assertEqual(
                action.action_scales,
                (0.0, 0.0, expected, expected, expected, expected),
            )

    def test_unregistered_zero_shot_scale_is_rejected(self) -> None:
        action = SimpleNamespace(action_scales=(0.0, 0.0, 0.07, 0.07, 0.07, 0.07))
        descriptor = evaluator.resolve_ablation("zero-shot-scale-0.035").to_dict()
        descriptor["deployment_leg_scale_rad"] = 0.05
        with self.assertRaisesRegex(ValueError, "not registered"):
            adapter.apply_environment_ablation(action, descriptor)

    def test_mode_always_on_forces_mode_and_detaches_live_trigger(self) -> None:
        action = SimpleNamespace(
            stair_trigger_sensor_name="stair_trigger_contact",
            stair_mode_forced=False,
        )
        adapter.apply_environment_ablation(
            action, evaluator.resolve_ablation("mode-always-on").to_dict()
        )
        self.assertIsNone(action.stair_trigger_sensor_name)
        self.assertTrue(action.stair_mode_forced)

    def test_leg_off_rejects_changed_indices(self) -> None:
        descriptor = evaluator.resolve_ablation("leg-off").to_dict()
        descriptor["zero_action_indices"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "2..5"):
            adapter.apply_policy_ablation(lambda value: value, descriptor)


class PreResetEvidenceTest(unittest.TestCase):
    def test_same_step_mode_rising_and_termination_survives_auto_reset(self) -> None:
        mode = torch.zeros(1, dtype=torch.bool)
        contact = torch.zeros(1, dtype=torch.bool)
        termination_contact = torch.zeros(1, dtype=torch.bool)
        root_x = torch.zeros(1)
        root_pos = torch.zeros((1, 3))

        class ActionManager:
            @staticmethod
            def get_term(name):
                if name != "hybrid_wheel_leg":
                    raise KeyError(name)
                return SimpleNamespace(stair_mode=mode)

        class TerminationManager:
            @staticmethod
            def get_term(name):
                if name != "non_wheel_ground_contact":
                    raise KeyError(name)
                return termination_contact

        def contact_func(_env, sensor_name):
            self.assertEqual(sensor_name, "non_wheel")
            return contact

        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            action_manager=ActionManager(),
            termination_manager=TerminationManager(),
            reset_terminated=torch.zeros(1, dtype=torch.bool),
            reset_buf=torch.zeros(1, dtype=torch.bool),
            scene={
                "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))
            },
        )
        cfg = SimpleNamespace(
            params={
                "non_wheel_contact_func": contact_func,
                "sensor_name": "non_wheel",
            }
        )
        evidence = adapter._LivePreResetEvidenceMetric(cfg, env)

        class Wrapped:
            unwrapped = env

            @staticmethod
            def reset():
                mode[:] = False
                contact[:] = False
                termination_contact[:] = False
                env.reset_terminated[:] = False
                env.reset_buf[:] = False
                evidence.reset()
                return {}, {}

            @staticmethod
            def step(_actions):
                # This is the state at MjLab's post-reward/pre-reset metric call.
                mode[:] = True
                contact[:] = True
                termination_contact[:] = True
                env.reset_terminated[:] = True
                env.reset_buf[:] = True
                root_x[:] = 0.25
                root_pos[:, 0] = 0.25
                evidence(env, contact_func, "non_wheel")
                # Simulate ActionManager.reset and root reset before step returns.
                mode[:] = False
                contact[:] = False
                termination_contact[:] = False
                root_x[:] = -3.0
                root_pos[:, 0] = -3.0
                return {}, torch.zeros(1), torch.ones(1), {}

        tracker = adapter._SafetyTrackingWrapper(Wrapped(), SimpleNamespace())
        tracker.reset()
        tracker.step(torch.zeros(1, 6))

        self.assertFalse(bool(mode[0]))  # Post-reset state really lost the latch.
        self.assertTrue(bool(evidence.last_mode_rising[0]))
        self.assertTrue(bool(evidence.last_terminated[0]))
        self.assertTrue(bool(evidence.last_contact[0]))
        self.assertEqual(float(evidence.last_root_x[0]), 0.25)
        self.assertEqual(
            (
                tracker.counts.events,
                tracker.counts.terminations,
                tracker.counts.non_wheel_contacts,
                tracker.counts.stair_mode_false_positives,
            ),
            (1, 1, 1, 1),
        )


class MockedRolloutTest(unittest.TestCase):
    def test_smoke_scan_accounts_settle_drive_stable_trials(self) -> None:
        config = _config("stairs")
        backend = FakeBackend()
        collection = adapter.collect_with_backend(config, backend)
        self.assertEqual(len(backend.scans), 1)
        request = backend.scans[0]
        self.assertEqual(
            (
                request.cells,
                request.num_envs_per_cell,
                request.repeats,
                request.settle_steps,
                request.drive_steps,
                request.stable_steps,
            ),
            ((0.01,), 1, 1, 2, 5, 2),
        )
        self.assertEqual(
            collection["rows"],
            [
                {
                    "height_m": 0.01,
                    "trials": 1,
                    "successes": 1,
                    "terminations": 0,
                    "non_wheel_contacts": 0,
                    "stair_mode_false_positives": 0,
                    "trigger_count": 1,
                    "pre_impact_trigger_count": 0,
                }
            ],
        )
        finalized = evaluator.finalize_adapter_output(config, collection)
        self.assertEqual(
            finalized["checkpoint"]["training"]["task"], STAIR_CAMP_TASK_ID
        )

    def test_mocked_failure_is_counted_once_per_trial(self) -> None:
        backend = FakeBackend()
        backend.fail_first = True
        collection = adapter.collect_with_backend(_config("slope"), backend)
        row = collection["rows"][0]
        self.assertEqual(row["successes"], 0)
        self.assertEqual(row["terminations"], 1)
        self.assertEqual(row["non_wheel_contacts"], 1)
        self.assertEqual(row["stair_mode_false_positives"], 1)

    def test_formal_stairs_request_has_exact_registered_grid_and_48_trials(
        self,
    ) -> None:
        config = _config("stairs", profile="formal")
        backend = FakeBackend()
        collection = adapter.collect_with_backend(config, backend)
        request = backend.scans[0]
        self.assertEqual(request.cells, evaluator.STAIR_HEIGHTS_M)
        self.assertEqual(request.events_per_cell, 48)
        self.assertEqual(len(collection["rows"]), 7)
        self.assertTrue(all(row["trials"] == 48 for row in collection["rows"]))

    def test_duplicate_and_missing_trial_identities_fail_closed(self) -> None:
        request = adapter._scan_request(_config("stairs"))
        trial = _trial(0.01, 1, 0)
        with self.assertRaisesRegex(ValueError, "expected exactly"):
            adapter.aggregate_scan_trials(request, [])
        formal_request = adapter.ScanRequest(
            domain="stairs",
            profile="smoke",
            terrain="pyramid_stairs",
            cell_key="height_m",
            cells=(0.01,),
            num_envs_per_cell=2,
            repeats=1,
            settle_steps=2,
            drive_steps=5,
            stable_steps=2,
            travel_distance_m=0.4,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapter.aggregate_scan_trials(formal_request, [trial, dict(trial)])

    def test_all_four_gate_bindings_reach_backend_unchanged(self) -> None:
        config = _config("flat")
        backend = FakeBackend()
        collection = adapter.collect_with_backend(config, backend)
        self.assertEqual(
            tuple(request.name for request in backend.gates), evaluator.GATE_NAMES
        )
        self.assertEqual(
            [(request.num_envs, request.steps) for request in backend.gates],
            [(1, 7), (1, 5), (1, 5), (1, 5)],
        )
        self.assertEqual(backend.gates[-1].minimum_kick_events, 1)
        result = evaluator.finalize_adapter_output(config, collection)
        self.assertTrue(result["all_gates_passed"])
        for row in result["gates"]:
            self.assertEqual(row["terminations"], 0)
            self.assertEqual(row["non_wheel_contacts"], 0)
            self.assertEqual(row["stair_mode_false_positives"], 0)

    def test_gate_outcome_cannot_forge_protocol_counts(self) -> None:
        class BadBackend(FakeBackend):
            def run_gate(self, request):
                result = super().run_gate(request)
                result["steps"] += 1
                return result

        with self.assertRaisesRegex(ValueError, "steps"):
            adapter.collect_with_backend(_config("flat"), BadBackend())

    def test_collect_constructs_real_backend_only_after_validation(self) -> None:
        config = _config("stairs")
        backend = FakeBackend()
        with (
            patch.object(
                adapter, "_load_live_dependencies", return_value="deps"
            ) as load,
            patch.object(adapter, "_MjLabBackend", return_value=backend) as backend_cls,
        ):
            adapter.collect(config)
        load.assert_called_once_with()
        backend_cls.assert_called_once_with(config, "deps")

        bad = copy.deepcopy(config)
        bad["task"] = "HopperTrex-Hybrid-v2-Stage5"
        with (
            patch.object(adapter, "_load_live_dependencies") as forbidden,
            self.assertRaises(ValueError),
        ):
            adapter.collect(bad)
        forbidden.assert_not_called()


class OutputPublicationTest(unittest.TestCase):
    def test_existing_target_and_fsync_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            output.write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                adapter._write_output({"winner": 1}, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "old")
            self.assertEqual(
                list(Path(temporary).glob(".result.json.incomplete.*")), []
            )

            output.unlink()
            with patch.object(adapter.os, "fsync", wraps=adapter.os.fsync) as fsync:
                adapter._write_output({"winner": 2}, output)
            self.assertTrue(fsync.called)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), {"winner": 2}
            )
            self.assertEqual(
                list(Path(temporary).glob(".result.json.incomplete.*")), []
            )

    def test_concurrent_publishers_have_one_hard_link_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "concurrent.json"
            barrier = threading.Barrier(3)
            successes: list[int] = []
            errors: list[BaseException] = []

            def publish(value: int) -> None:
                try:
                    barrier.wait(timeout=5.0)
                    adapter._write_output({"winner": value}, output)
                    successes.append(value)
                except BaseException as exc:  # noqa: BLE001 - assertion capture.
                    errors.append(exc)

            threads = [
                threading.Thread(target=publish, args=(1,)),
                threading.Thread(target=publish, args=(2,)),
            ]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5.0)
            for thread in threads:
                thread.join(timeout=5.0)

            self.assertEqual(len(successes), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileExistsError)
            self.assertIn(
                json.loads(output.read_text(encoding="utf-8"))["winner"], (1, 2)
            )
            self.assertEqual(
                list(Path(temporary).glob(".concurrent.json.incomplete.*")), []
            )


class TriggerFalsePositiveTest(unittest.TestCase):
    def test_exact_mapping_matches_preflight_schema_for_both_domains(self) -> None:
        from hoppertrex_mjlab.scripts.rsl_rl.preflight_stair_camp import (
            validate_live_false_positive_result,
        )

        for domain, events in (
            ("camp_flat_rolling", 96_000),
            ("stage5_kick", 128),
        ):
            with self.subTest(domain=domain):
                result = adapter.make_trigger_false_positive_check(
                    domain=domain,
                    events=events,
                    stair_mode_false_positives=0,
                )
                self.assertEqual(
                    set(result),
                    {
                        "schema_version",
                        "kind",
                        "domain",
                        "threshold_n",
                        "window_steps",
                        "events",
                        "stair_mode_false_positives",
                        "completed",
                    },
                )
                self.assertEqual(result["threshold_n"], 18.0)
                self.assertEqual(result["window_steps"], 3)
                validated = validate_live_false_positive_result(
                    result, expected_domain=domain
                )
                self.assertTrue(validated["passed"])

    def test_pretraining_request_is_exact_and_contains_no_checkpoint(self) -> None:
        request = _pretraining_request()
        self.assertEqual(
            _CONTRACT_SHA,
            "1d4b18db32e48b3ae8803e385a032203bdddc7f8198da9679f519bc8947190cb",
        )
        self.assertEqual(
            set(request),
            {
                "schema_version",
                "kind",
                "task",
                "evaluation_seed",
                "device",
                "git_sha",
                "contract_sha256",
                "artifact_bindings",
            },
        )
        self.assertEqual(
            set(request["artifact_bindings"]),
            set(adapter.PRETRAINING_ARTIFACT_BINDING_NAMES),
        )
        self.assertNotIn("checkpoint", json.dumps(request).lower())
        self.assertEqual(adapter.validate_pretraining_trigger_request(request), request)

    def test_pretraining_request_mutations_are_rejected(self) -> None:
        mutations: list[tuple[str, dict[str, object]]] = []

        def add(name: str, mutate) -> None:
            candidate = copy.deepcopy(_pretraining_request())
            mutate(candidate)
            mutations.append((name, candidate))

        add("extra top-level", lambda value: value.__setitem__("checkpoint", {}))
        add("missing top-level", lambda value: value.pop("git_sha"))
        add("bool schema", lambda value: value.__setitem__("schema_version", True))
        add("kind", lambda value: value.__setitem__("kind", "other"))
        add("task", lambda value: value.__setitem__("task", "Stage5"))
        add("seed", lambda value: value.__setitem__("evaluation_seed", 2))
        add("device", lambda value: value.__setitem__("device", " cpu"))
        add("git", lambda value: value.__setitem__("git_sha", "A" * 40))
        add(
            "contract",
            lambda value: value.__setitem__("contract_sha256", "b" * 64),
        )
        add(
            "missing artifact",
            lambda value: value["artifact_bindings"].pop("calibration_hash"),
        )
        add(
            "extra artifact",
            lambda value: value["artifact_bindings"].__setitem__(
                "unregistered_hash", "7" * 64
            ),
        )
        add(
            "artifact digest",
            lambda value: value["artifact_bindings"].__setitem__(
                "calibration_hash", "F" * 64
            ),
        )

        for name, candidate in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.validate_pretraining_trigger_request(candidate)

    def test_both_live_fp_domains_use_their_fixed_formal_binding(self) -> None:
        backend = FakeBackend()
        request = _pretraining_request()
        flat = adapter.collect_trigger_false_positive_with_backend(
            {"domain": "camp_flat_rolling", "pretraining_request": request},
            backend,
        )
        kick = adapter.collect_trigger_false_positive_with_backend(
            {"domain": "stage5_kick", "pretraining_request": request},
            backend,
        )
        self.assertEqual(flat["events"], 96_000)
        self.assertEqual(kick["events"], 128)
        self.assertEqual(
            [
                (
                    domain,
                    gate.name,
                    gate.profile,
                    gate.num_envs,
                    gate.steps,
                    gate.minimum_kick_events,
                )
                for domain, gate in backend.fp
            ],
            [
                (
                    "camp_flat_rolling",
                    "velocity_gate_passed",
                    "formal",
                    16,
                    3000,
                    0,
                ),
                (
                    "stage5_kick",
                    "stage5_gate_passed",
                    "formal",
                    32,
                    3000,
                    128,
                ),
            ],
        )

    def test_fp_helper_rejects_checkpoint_shape_and_bad_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema drifted"):
            adapter.collect_trigger_false_positive_with_backend(
                {"domain": "stage5_kick", "adapter_config": _config("flat")},
                FakeBackend(),
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            adapter.make_trigger_false_positive_check(
                domain="stage5_kick",
                events=1,
                stair_mode_false_positives=2,
            )

    def test_fp_backend_schema_and_formal_event_count_are_strict(self) -> None:
        class ExtraBackend(FakeBackend):
            def run_trigger_false_positive(self, domain, request):
                del domain, request
                return {
                    "events": 128,
                    "stair_mode_false_positives": 0,
                    "extra": 0,
                }

        class ShortBackend(FakeBackend):
            def run_trigger_false_positive(self, domain, request):
                del domain, request
                return {"events": 1, "stair_mode_false_positives": 0}

        envelope = {
            "domain": "stage5_kick",
            "pretraining_request": _pretraining_request(),
        }
        with self.assertRaisesRegex(ValueError, "schema drifted"):
            adapter.collect_trigger_false_positive_with_backend(
                envelope, ExtraBackend()
            )
        with self.assertRaisesRegex(ValueError, "formal binding"):
            adapter.collect_trigger_false_positive_with_backend(
                envelope, ShortBackend()
            )

    def test_runtime_registry_rejects_every_provenance_drift(self) -> None:
        adapter._PretrainingFpBackend(
            _pretraining_request(), _pretraining_dependencies()
        )

        with self.assertRaisesRegex(RuntimeError, "Git SHA"):
            adapter._PretrainingFpBackend(
                _pretraining_request(),
                _pretraining_dependencies(runtime_git="b" * 40),
            )
        with self.assertRaisesRegex(RuntimeError, "contract drifted"):
            adapter._PretrainingFpBackend(
                _pretraining_request(),
                _pretraining_dependencies(runtime_contract="b" * 64),
            )
        drifted_artifacts = dict(_ARTIFACTS)
        drifted_artifacts["controller_gain_hash"] = "9" * 64
        with self.assertRaisesRegex(RuntimeError, "artifacts drifted"):
            adapter._PretrainingFpBackend(
                _pretraining_request(),
                _pretraining_dependencies(runtime_artifacts=drifted_artifacts),
            )

        drifted_request = _pretraining_request()
        drifted_request["artifact_bindings"]["posture_map_hash"] = "8" * 64
        with self.assertRaisesRegex(RuntimeError, "artifacts drifted"):
            adapter._PretrainingFpBackend(drifted_request, _pretraining_dependencies())

        module_drift = _pretraining_dependencies()
        module_drift.contract_module.STAIR_CAMP_CANONICAL_CONTRACT_SHA256 = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "constant drifted"):
            adapter._PretrainingFpBackend(_pretraining_request(), module_drift)

    def test_flat_pretraining_rollout_uses_only_six_wide_zero_policy(self) -> None:
        backend = adapter._PretrainingFpBackend(
            _pretraining_request(), _pretraining_dependencies()
        )

        class Tracker:
            def __init__(self) -> None:
                self.counts = adapter._SafetyCounts()
                self.actions: list[torch.Tensor] = []

            def reset(self):
                return None

            def get_observations(self):
                return {
                    "actor": torch.ones(2, 52),
                    "critic": torch.ones(2, 55),
                }

            def step(self, actions):
                self.actions.append(actions.clone())
                self.counts.events += 2
                return None

        tracker = Tracker()
        gate = adapter.GateRequest(
            name="velocity_gate_passed",
            source_suite="hybrid_linear_velocity",
            terrain="flat",
            profile="smoke",
            num_envs=2,
            steps=3,
            scenario_count=2,
            commands=((-0.07, 0.0), (0.07, 0.0)),
            settle_steps=0,
            measure_steps=0,
            kick_scale=None,
            minimum_kick_events=0,
        )
        session = SimpleNamespace(tracker=tracker, env_cfg=SimpleNamespace())
        with (
            patch.object(
                backend,
                "_session",
                return_value=adapter.contextlib.nullcontext(session),
            ),
            patch.object(backend, "_posture_center", return_value=(0.0, 0.0)),
            patch.object(backend, "_force_commands"),
            patch.object(
                backend,
                "_load_policy",
                side_effect=AssertionError("checkpoint load forbidden"),
            ) as load_policy,
        ):
            outcome = backend.run_trigger_false_positive("camp_flat_rolling", gate)
        load_policy.assert_not_called()
        self.assertEqual(outcome, {"events": 12, "stair_mode_false_positives": 0})
        self.assertEqual(len(tracker.actions), 6)
        for actions in tracker.actions:
            self.assertEqual(tuple(actions.shape), (2, 6))
            self.assertTrue(torch.equal(actions, torch.zeros_like(actions)))

    def test_stage5_pretraining_kick_uses_zero_policy_and_128_events(self) -> None:
        backend = adapter._PretrainingFpBackend(
            _pretraining_request(), _pretraining_dependencies()
        )
        gate = adapter._formal_pretraining_gate_requests()["stage5_gate_passed"]
        captured: list[torch.Tensor] = []

        def recovery(request, _base, *, policy, purpose):
            self.assertEqual(request.kick_scale, 8.0)
            self.assertEqual(request.minimum_kick_events, 128)
            self.assertEqual(purpose, "pretraining_stage5_kick_fp")
            actions = policy(
                {
                    "actor": torch.ones(32, 52),
                    "critic": torch.ones(32, 55),
                }
            )
            captured.append(actions)
            return {"kick_event_count": 128.0}, adapter._SafetyCounts()

        with (
            patch.object(backend, "_recovery_metrics", side_effect=recovery),
            patch.object(
                backend,
                "_load_policy",
                side_effect=AssertionError("checkpoint load forbidden"),
            ) as load_policy,
        ):
            outcome = backend.run_trigger_false_positive("stage5_kick", gate)
        load_policy.assert_not_called()
        self.assertEqual(outcome, {"events": 128, "stair_mode_false_positives": 0})
        self.assertEqual(len(captured), 1)
        self.assertEqual(tuple(captured[0].shape), (32, 6))
        self.assertTrue(torch.equal(captured[0], torch.zeros_like(captured[0])))

    def test_trigger_fp_cli_uses_request_file_shape(self) -> None:
        args = adapter.parse_args(
            [
                "trigger-fp",
                "--domain",
                "stage5_kick",
                "--request",
                "pretraining-request.json",
            ]
        )
        self.assertEqual(args.command, "trigger-fp")
        self.assertEqual(args.domain, "stage5_kick")
        self.assertEqual(args.request, Path("pretraining-request.json"))
        self.assertFalse(hasattr(args, "config"))

        with (
            patch.object(
                adapter, "_read_json_mapping", return_value=_pretraining_request()
            ),
            patch.object(
                adapter,
                "collect_trigger_false_positive_check",
                return_value={"completed": True},
            ) as collect_check,
            patch.object(adapter, "_write_output"),
        ):
            self.assertEqual(
                adapter.main(
                    [
                        "trigger-fp",
                        "--domain",
                        "stage5_kick",
                        "--request",
                        "pretraining-request.json",
                    ]
                ),
                0,
            )
        collect_check.assert_called_once_with(
            {
                "domain": "stage5_kick",
                "pretraining_request": _pretraining_request(),
            }
        )


if __name__ == "__main__":
    unittest.main()

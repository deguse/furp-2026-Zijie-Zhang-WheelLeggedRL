#!/usr/bin/env python3
'''Run candidate 24 through the C1 affine 15-cell final gate.'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from hoppertrex_mjlab import tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.controller_schedule import (
  SELECTION_METRICS,
)
from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_center_smoke import (
  load_affine_nodes,
)
from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate import (
  EXPECTED_BINDINGS,
  MINIMUM_INCUMBENT_COMMAND_GAIN_RATIO,
  NODE_STEMS,
  aggregate_candidate,
  candidate_schedule,
  evaluation_cells,
  load_registered_floors,
  registered_caps,
  run_cell,
)

FORMAL_TASK = 'HopperTrex-Hybrid-v2-Stage3'
FORMAL_DEVICE = 'cuda:0'
FORMAL_NUM_ENVS = 16
FORMAL_SETTLE_STEPS = 100
FORMAL_MEASURE_STEPS = 200
FORMAL_VX_CHECK = 0.05
FORMAL_SEED = 1

SOURCE_COLLECTION_GIT_SHA = '0c7bd78893998f0a1c6d58615fb3ea7fd97f0bdd'
SOURCE_COLLECTION_ZIP_SHA256 = (
  '10e0f8f498107406e969e9f7d8390f8ac8c22f5838b60d5254e65196453eb4f9'
)
RETRY_EVALUATOR_GIT_SHA = '9fe48c31a5cc1c3cbea8b163d3fafe860e3aba53'
RETRY_RESULT_SHA256 = (
  '18cea95353b227b47370af25265f16c2450ba25e224069e08c52f92d6d472f07'
)
RETRY_PROTOCOL_SHA256 = (
  'c336eb937a12252412bb2a8837504eaf13635fb5a0f3a4da2d33a1a1443b5c98'
)
RETRY_ZIP_SHA256 = (
  '86521c7e5762b669a2c179c590f5c08fbd6454d165087ee8a02b86ae293f14dd'
)
EXPECTED_MJLAB_GIT_SHA = '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'
SELECTED_CANDIDATE_INDEX = 24
SELECTED_Q_DIAG = [40.0, 4.0, 8.0, 1.0]
SELECTED_R_DIAG = [0.5]
SELECTED_ANCHOR_ALPHA = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--nodes-dir', type=Path, required=True)
  parser.add_argument('--source-zip', type=Path, required=True)
  parser.add_argument('--retry-result', type=Path, required=True)
  parser.add_argument('--retry-protocol', type=Path, required=True)
  parser.add_argument('--retry-zip', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--compensated-qualification', type=Path, required=True)
  parser.add_argument('--git-sha', required=True)
  parser.add_argument('--mjlab-git-sha', required=True)
  parser.add_argument('--task', default=FORMAL_TASK)
  parser.add_argument('--device', default=FORMAL_DEVICE)
  parser.add_argument('--num-envs', type=int, default=FORMAL_NUM_ENVS)
  parser.add_argument('--settle-steps', type=int, default=FORMAL_SETTLE_STEPS)
  parser.add_argument('--measure-steps', type=int, default=FORMAL_MEASURE_STEPS)
  parser.add_argument('--vx-check', type=float, default=FORMAL_VX_CHECK)
  return parser.parse_args(argv)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
  payload = json.loads(path.read_text(encoding='utf-8-sig'))
  if not isinstance(payload, dict):
    raise TypeError(f'JSON root must be an object: {path}')
  return payload


def _require_sha(path: Path, expected: str, label: str) -> None:
  if not path.is_file():
    raise FileNotFoundError(f'Missing {label}: {path}')
  actual = _sha256(path)
  if actual != expected:
    raise ValueError(f'{label} SHA256 mismatch: expected {expected}, got {actual}.')


def _validate_formal_args(args: argparse.Namespace) -> None:
  expected = {
    'task': FORMAL_TASK,
    'device': FORMAL_DEVICE,
    'num_envs': FORMAL_NUM_ENVS,
    'settle_steps': FORMAL_SETTLE_STEPS,
    'measure_steps': FORMAL_MEASURE_STEPS,
    'vx_check': FORMAL_VX_CHECK,
    'mjlab_git_sha': EXPECTED_MJLAB_GIT_SHA,
  }
  for name, value in expected.items():
    if getattr(args, name) != value:
      raise ValueError(f'Formal argument --{name.replace("_", "-")} is frozen.')
  if re.fullmatch(r'[0-9a-f]{40}', args.git_sha) is None:
    raise ValueError('--git-sha must be a full lowercase Git SHA.')


def load_selected_evidence(
  *, source_zip: Path, retry_result: Path, retry_protocol: Path, retry_zip: Path
) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
  '''Verify the frozen retry and recover only its preregistered winner.'''

  _require_sha(source_zip, SOURCE_COLLECTION_ZIP_SHA256, 'source collection ZIP')
  _require_sha(retry_result, RETRY_RESULT_SHA256, 'retry result')
  _require_sha(retry_protocol, RETRY_PROTOCOL_SHA256, 'retry protocol')
  _require_sha(retry_zip, RETRY_ZIP_SHA256, 'retry ZIP')
  payload = _read_json(retry_result)
  protocol = _read_json(retry_protocol)
  required_payload = {
    'classification': 'AFFINE_CENTER_SMOKE_HAS_CANDIDATES',
    'git_sha': RETRY_EVALUATOR_GIT_SHA,
    'collection_git_sha': SOURCE_COLLECTION_GIT_SHA,
    'mjlab_git_sha': EXPECTED_MJLAB_GIT_SHA,
    'passed_candidate_count': 27,
    'completed_candidate_count': 27,
    'completed_node_fit_count': 243,
    'evidence_eligible': True,
    'promotion_eligible': False,
    'training_eligible': False,
    'checkpoint': None,
    'next_step': 'DOWNLOAD_FOR_REVIEW',
  }
  for key, expected in required_payload.items():
    if payload.get(key) != expected:
      raise ValueError(f'Retry result field {key} is not frozen as expected.')
  required_protocol = {
    'kind': 'c1_affine_center_smoke_retry',
    'git_sha': RETRY_EVALUATOR_GIT_SHA,
    'source_collection_git_sha': SOURCE_COLLECTION_GIT_SHA,
    'source_zip_sha256': SOURCE_COLLECTION_ZIP_SHA256,
    'mjlab_git_sha': EXPECTED_MJLAB_GIT_SHA,
    'passed_candidate_count': 27,
    'classification': 'AFFINE_CENTER_SMOKE_HAS_CANDIDATES',
  }
  for key, expected in required_protocol.items():
    if protocol.get(key) != expected:
      raise ValueError(f'Retry protocol field {key} is not frozen as expected.')
  if payload.get('bindings') != EXPECTED_BINDINGS:
    raise ValueError('Retry artifact bindings do not match the frozen C1 stack.')
  candidates = payload.get('candidates')
  if not isinstance(candidates, list) or len(candidates) != 27:
    raise ValueError('Retry result must contain all 27 candidates.')
  passed = [item for item in candidates if item.get('flat_gate_passed') is True]
  if len(passed) != 27:
    raise ValueError('Retry result no longer records 27/27 passing candidates.')
  selected = min(
    passed,
    key=lambda item: tuple(float(item[metric]) for metric in SELECTION_METRICS),
  )
  if selected.get('index') != SELECTED_CANDIDATE_INDEX:
    raise ValueError('Frozen lexicographic selection is no longer candidate 24.')
  if selected.get('q_diag') != SELECTED_Q_DIAG:
    raise ValueError('Candidate 24 Q diagonal drifted.')
  if selected.get('r_diag') != SELECTED_R_DIAG:
    raise ValueError('Candidate 24 R diagonal drifted.')
  if not math.isclose(
    float(selected.get('anchor_alpha', math.nan)),
    SELECTED_ANCHOR_ALPHA,
    rel_tol=0.0,
    abs_tol=1.0e-15,
  ):
    raise ValueError('Candidate 24 anchor alpha drifted.')
  node_facts = selected.get('node_facts')
  if not isinstance(node_facts, dict) or set(node_facts) != set(NODE_STEMS):
    raise ValueError('Candidate 24 must contain exactly nine node facts.')
  gains = np.zeros((3, 3, 4), dtype=np.float64)
  for stem in NODE_STEMS:
    fact = node_facts[stem]
    if (
      fact.get('controllability_rank') != 4
      or float(fact.get('max_nrmse', math.inf)) > 0.15
      or fact.get('fallback_reasons') != []
      or float(fact.get('command_gain_ratio', -math.inf))
      < MINIMUM_INCUMBENT_COMMAND_GAIN_RATIO
    ):
      raise ValueError(f'Candidate 24 node {stem} is not qualified.')
    gain = np.asarray(fact.get('gain'), dtype=np.float64)
    if gain.shape != (4,) or not np.all(np.isfinite(gain)):
      raise ValueError(f'Candidate 24 node {stem} gain is invalid.')
    gains[int(stem[6]), int(stem[9])] = gain
  return payload, selected, gains


def equilibrium_grids(
  nodes: dict[str, dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
  state = np.zeros((3, 3, 4), dtype=np.float64)
  control = np.zeros((3, 3), dtype=np.float64)
  for stem in NODE_STEMS:
    metadata = nodes[stem]['metadata']
    state[int(stem[6]), int(stem[9])] = metadata['equilibrium_state']
    control[int(stem[6]), int(stem[9])] = float(metadata['equilibrium_input'][0])
  return state, control


def make_payload(
  *,
  git_sha: str,
  retry_payload: dict[str, object],
  selected: dict[str, object],
  caps: dict[str, float],
  floors: dict[str, float],
  cells: list[dict[str, float]],
) -> dict[str, object]:
  verdict = aggregate_candidate(cells, caps)
  passed = bool(verdict['flat_gate_passed'])
  return {
    'schema_version': 1,
    'kind': 'c1_affine_full_gate',
    'classification': (
      'C1_AFFINE_FULL_GATE_SELECTED'
      if passed
      else 'C1_AFFINE_FULL_GATE_FAILED_STOP'
    ),
    'git_sha': git_sha,
    'collection_git_sha': SOURCE_COLLECTION_GIT_SHA,
    'retry_evaluator_git_sha': RETRY_EVALUATOR_GIT_SHA,
    'mjlab_git_sha': EXPECTED_MJLAB_GIT_SHA,
    'source_collection_zip_sha256': SOURCE_COLLECTION_ZIP_SHA256,
    'retry_result_sha256': RETRY_RESULT_SHA256,
    'retry_protocol_sha256': RETRY_PROTOCOL_SHA256,
    'retry_zip_sha256': RETRY_ZIP_SHA256,
    'bindings': retry_payload['bindings'],
    'candidate': {
      'index': selected['index'],
      'q_diag': selected['q_diag'],
      'r_diag': selected['r_diag'],
      'anchor_alpha': selected['anchor_alpha'],
      'selection_metrics': {
        metric: selected[metric] for metric in SELECTION_METRICS
      },
      'node_facts': selected['node_facts'],
    },
    'run_protocol': {
      'task': FORMAL_TASK,
      'device': FORMAL_DEVICE,
      'seed': FORMAL_SEED,
      'num_envs': FORMAL_NUM_ENVS,
      'settle_steps': FORMAL_SETTLE_STEPS,
      'measure_steps': FORMAL_MEASURE_STEPS,
      'vx_check': FORMAL_VX_CHECK,
      'candidate_count': 1,
      'cell_count': len(cells),
      'cell_order': [
        {
          'height_m': cell['target_height'],
          'pitch_rad': cell['target_pitch'],
          'vx_m_s': cell['vx_command'],
        }
        for cell in cells
      ],
      'reset_before_each_cell': True,
      'scope': 'single_selected_c1_final_flat_gate',
    },
    'floors': floors,
    'caps': caps,
    'cells': cells,
    **verdict,
    'completed_candidate_count': 1,
    'completed_cell_count': len(cells),
    'evidence_eligible': True,
    'promotion_eligible': False,
    'training_eligible': False,
    'checkpoint': None,
    'yaw_calibration_hash': None,
    'next_step': 'DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD' if passed else 'STOP',
  }


def make_selection_payload(
  *,
  detail: dict[str, object],
  retry_payload: dict[str, object],
  detail_path: Path,
) -> dict[str, object]:
  if detail['classification'] != 'C1_AFFINE_FULL_GATE_SELECTED':
    raise ValueError('Selection evidence requires a passing full gate.')
  screened = [
    {
      'index': candidate['index'],
      'q_diag': candidate['q_diag'],
      'r_diag': candidate['r_diag'],
      'anchor_alpha': candidate['anchor_alpha'],
      'center_smoke_passed': candidate['flat_gate_passed'],
      **{metric: candidate[metric] for metric in SELECTION_METRICS},
    }
    for candidate in retry_payload['candidates']
  ]
  return {
    'schema_version': 1,
    'kind': 'c1_affine_full_gate_selection',
    'status': 'affine_full_gate_selected',
    'classification': detail['classification'],
    'git_sha': detail['git_sha'],
    'mjlab_git_sha': detail['mjlab_git_sha'],
    'full_gate_artifact_path': detail_path.name,
    'full_gate_artifact_sha256': _sha256(detail_path),
    'source_collection_zip_sha256': SOURCE_COLLECTION_ZIP_SHA256,
    'retry_result_sha256': RETRY_RESULT_SHA256,
    'retry_protocol_sha256': RETRY_PROTOCOL_SHA256,
    'retry_zip_sha256': RETRY_ZIP_SHA256,
    'screened_candidates': screened,
    'selected_candidate_index': SELECTED_CANDIDATE_INDEX,
    'final_gate_candidate': {
      'index': SELECTED_CANDIDATE_INDEX,
      'q_diag': SELECTED_Q_DIAG,
      'r_diag': SELECTED_R_DIAG,
      'anchor_alpha': SELECTED_ANCHOR_ALPHA,
      'flat_gate_passed': detail['flat_gate_passed'],
      'safety_clean': detail['safety_clean'],
      **{metric: detail[metric] for metric in SELECTION_METRICS},
    },
    'evidence_eligible': True,
    'promotion_eligible': False,
    'training_eligible': False,
    'checkpoint': None,
    'yaw_calibration_hash': None,
  }


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  _validate_formal_args(args)
  output = args.output.resolve()
  selection_output = output.with_name('c1_affine_full_gate_selection.json')
  if output.exists() or selection_output.exists():
    raise FileExistsError(
      f'Refusing to overwrite C1 affine full-gate output beside: {output}'
    )
  if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('Formal C1 affine full gate requires CUDA device 0.')
  nodes = load_affine_nodes(args.nodes_dir.resolve())
  collection_shas = {
    str(node['metadata'].get('git_sha')) for node in nodes.values()
  }
  if collection_shas != {SOURCE_COLLECTION_GIT_SHA}:
    raise ValueError('Nine-node collection Git provenance drifted.')
  retry_payload, selected, gains = load_selected_evidence(
    source_zip=args.source_zip.resolve(),
    retry_result=args.retry_result.resolve(),
    retry_protocol=args.retry_protocol.resolve(),
    retry_zip=args.retry_zip.resolve(),
  )
  equilibrium_state, equilibrium_input = equilibrium_grids(nodes)
  cfg = load_env_cfg(args.task, play=True)
  cfg.seed = FORMAL_SEED
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  action_cfg = cfg.actions['hybrid_wheel_leg']
  runtime_bindings = {
    'controller_gain_hash': action_cfg.controller_gain_hash,
    'velocity_calibration_hash': action_cfg.calibration_hash,
    'posture_artifact_hash': action_cfg.posture_artifact_hash,
    'station_calibration_hash': action_cfg.station_calibration_hash,
  }
  if runtime_bindings != EXPECTED_BINDINGS:
    raise ValueError('Loaded runtime bindings do not match candidate 24 evidence.')
  floors = load_registered_floors(
    args.compensated_qualification.resolve(),
    expected_controller_gain_hash=action_cfg.controller_gain_hash,
  )
  caps = registered_caps(floors)
  action_cfg.controller_schedule = candidate_schedule(
    gains,
    equilibrium_state,
    SELECTED_Q_DIAG,
    SELECTED_R_DIAG,
    equilibrium_input,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    cells = [
      run_cell(
        env,
        height=height,
        pitch=pitch,
        vx=vx,
        settle_steps=args.settle_steps,
        measure_steps=args.measure_steps,
      )
      for height, pitch, vx in evaluation_cells(args.vx_check)
    ]
  finally:
    env.close()
  payload = make_payload(
    git_sha=args.git_sha,
    retry_payload=retry_payload,
    selected=selected,
    caps=caps,
    floors=floors,
    cells=cells,
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
    encoding='utf-8',
  )
  if payload['classification'] == 'C1_AFFINE_FULL_GATE_SELECTED':
    selection = make_selection_payload(
      detail=payload,
      retry_payload=retry_payload,
      detail_path=output,
    )
    selection_output.write_text(
      json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
    )
  print(
    f'[c1-full-gate] classification={payload["classification"]} '
    f'candidate={SELECTED_CANDIDATE_INDEX} cells={len(cells)}'
  )


if __name__ == '__main__':
  main()

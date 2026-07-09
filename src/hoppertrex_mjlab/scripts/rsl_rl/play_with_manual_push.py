#!/usr/bin/env python3
"""Play HopperTrex policies with project-local manual push controls."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import mjlab
import torch
import tyro

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: F401
except ImportError:
  import tasks  # noqa: F401
import mjlab.scripts.play as mjlab_play
from mjlab.scripts.play import PlayConfig
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import UpdateReason, ViserPlayViewer

DEFAULT_TASK = "Mjlab-HopperTrex-Balance-v0"


class ManualPushViserPlayViewer(ViserPlayViewer):
  """Viser viewer with HopperTrex demo-only root velocity kick buttons."""

  def setup(self) -> None:
    super().setup()
    with self._server.gui.add_folder("Manual Push"):
      light_buttons = self._server.gui.add_button_group(
        "Light velocity kick",
        options=["+X", "-X", "+Pitch", "-Pitch"],
      )

      @light_buttons.on_click
      def _(event) -> None:
        self.request_action("CUSTOM", _push_payload(event.target.value, light=True))

      strong_buttons = self._server.gui.add_button_group(
        "Strong velocity kick",
        options=["+X", "-X", "+Pitch", "-Pitch"],
      )

      @strong_buttons.on_click
      def _(event) -> None:
        self.request_action("CUSTOM", _push_payload(event.target.value, light=False))

  def _handle_custom_action(
    self,
    action: ViewerAction,
    payload: Optional[Any],
  ) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "manual_push":
      self._handle_manual_push(
        x=float(payload.get("x", 0.0)),
        pitch=float(payload.get("pitch", 0.0)),
      )
      return True
    return super()._handle_custom_action(action, payload)

  def _handle_manual_push(self, x: float, pitch: float) -> None:
    """Apply a one-shot root velocity kick to the selected environment."""
    env = self.env.unwrapped
    try:
      asset = env.scene["robot"]
    except KeyError:
      self._last_error = "Manual Push requires an entity named 'robot'."
      print(f"[WARN] {self._last_error}")
      return

    env_ids = torch.tensor([self._scene.env_idx], dtype=torch.int64, device=env.device)
    with self._sim_lock:
      vel_w = asset.data.root_link_vel_w[env_ids].clone()
      vel_w[:, 0] += x
      vel_w[:, 4] += pitch
      asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
      env.sim.forward()
      env.sim.sense()

    self._pending_update_reasons.add(UpdateReason.ACTION)
    self._scene.request_update()


def _push_payload(label: str, light: bool) -> dict[str, float | str]:
  x_delta = 0.08 if light else 0.15
  pitch_delta = 0.12 if light else 0.25
  if label == "+X":
    return {"type": "manual_push", "x": x_delta, "pitch": 0.0}
  if label == "-X":
    return {"type": "manual_push", "x": -x_delta, "pitch": 0.0}
  if label == "+Pitch":
    return {"type": "manual_push", "x": 0.0, "pitch": pitch_delta}
  return {"type": "manual_push", "x": 0.0, "pitch": -pitch_delta}


def _normalize_argv() -> tuple[str, list[str]]:
  args = sys.argv[1:]
  task = DEFAULT_TASK
  if "--task" in args:
    idx = args.index("--task")
    task = args[idx + 1]
    args = args[:idx] + args[idx + 2 :]
  elif args and not args[0].startswith("-"):
    task = args[0]
    args = args[1:]
  return task, args


def main() -> None:
  task, remaining = _normalize_argv()
  default_cfg = replace(
    PlayConfig(),
    agent="zero",
    log_root=str(PROJECT_PATH / "logs" / "rsl_rl"),
    viewer="viser",
  )
  cfg = tyro.cli(
    PlayConfig,
    args=remaining,
    default=default_cfg,
    prog=f"{sys.argv[0]} {task}",
    config=mjlab.TYRO_FLAGS,
  )
  mjlab_play.ViserPlayViewer = ManualPushViserPlayViewer
  mjlab_play.run_play(task_id=task, cfg=cfg)


if __name__ == "__main__":
  main()

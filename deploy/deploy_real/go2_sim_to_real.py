#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import select
import sys
import threading
import time
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]

SDK_ROOT_CANDIDATES = (
    os.environ.get("UNITREE_SDK2_PYTHON_ROOT"),
    os.environ.get("UNITREE_SDK2PY_ROOT"),
    REPO_ROOT / "third_party" / "unitree_sdk2_python",
)


def _ensure_unitree_sdk_on_path() -> None:
    try:
        import unitree_sdk2py  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    for root in SDK_ROOT_CANDIDATES:
        if not root:
            continue
        candidate = Path(root).expanduser()
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import unitree_sdk2py  # noqa: F401
                return
            except ModuleNotFoundError:
                continue


_ensure_unitree_sdk_on_path()

MotionSwitcherClient = None
ChannelFactoryInitialize = None
ChannelPublisher = None
ChannelSubscriber = None
SportClient = None
unitree_go_msg_dds__LowCmd_ = None
LowCmd_ = None
LowState_ = None
CRC = None


def _load_unitree_sdk() -> None:
    global MotionSwitcherClient
    global ChannelFactoryInitialize
    global ChannelPublisher
    global ChannelSubscriber
    global SportClient
    global unitree_go_msg_dds__LowCmd_
    global LowCmd_
    global LowState_
    global CRC

    if MotionSwitcherClient is not None:
        return

    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient as _MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize as _ChannelFactoryInitialize
        from unitree_sdk2py.core.channel import ChannelPublisher as _ChannelPublisher
        from unitree_sdk2py.core.channel import ChannelSubscriber as _ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient as _SportClient
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as _unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as _LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as _LowState_
        from unitree_sdk2py.utils.crc import CRC as _CRC
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unitree SDK imports failed. Make sure both unitree_sdk2py and cyclonedds "
            "are available in the execution environment. If the SDK is checked out locally, "
            "set UNITREE_SDK2_PYTHON_ROOT to that directory."
        ) from exc

    MotionSwitcherClient = _MotionSwitcherClient
    ChannelFactoryInitialize = _ChannelFactoryInitialize
    ChannelPublisher = _ChannelPublisher
    ChannelSubscriber = _ChannelSubscriber
    SportClient = _SportClient
    unitree_go_msg_dds__LowCmd_ = _unitree_go_msg_dds__LowCmd_
    LowCmd_ = _LowCmd_
    LowState_ = _LowState_
    CRC = _CRC

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


DEFAULT_EXPERIMENT = "unitree_go2_deploy_baseline_lab_flat"
DEFAULT_RUN = None
DEFAULT_CHECKPOINT = -1
DEFAULT_POLICY = REPO_ROOT / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT / "exported" / "policy.pt"

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

POLICY_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

SDK_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

POLICY_TO_SDK_INDEX = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)

DEFAULT_JOINT_POS = np.array(
    [
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 1.0, -1.5,
        -0.1, 1.0, -1.5,
    ],
    dtype=np.float32,
)

DEFAULT_STARTUP_SIT_POSE_SDK = np.array(
    [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65],
    dtype=np.float32,
)
DEFAULT_STARTUP_STAND_POSE_SDK = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5],
    dtype=np.float32,
)

REMOTE_BUTTONS = (
    "r1",
    "l1",
    "start",
    "select",
    "r2",
    "l2",
    "f1",
    "f3",
    "a",
    "b",
    "x",
    "y",
    "up",
    "right",
    "down",
    "left",
)


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    name = str(name).lower()
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    if name == "selu":
        return nn.SELU
    if name == "lrelu":
        return nn.LeakyReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    raise ValueError(f"unsupported activation: {name}")


def _quat_to_gravity_body(q: Sequence[float], order: str = "wxyz") -> np.ndarray:
    if q is None or len(q) < 4:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    if order == "wxyz":
        w, x, y, z = [float(v) for v in q[:4]]
    else:
        x, y, z, w = [float(v) for v in q[:4]]

    n2 = w * w + x * x + y * y + z * z
    if n2 < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    inv_n = 1.0 / math.sqrt(n2)
    w, x, y, z = w * inv_n, x * inv_n, y * inv_n, z * inv_n
    gx = 2.0 * (x * z - y * w)
    gy = 2.0 * (y * z + x * w)
    gz = 1.0 - 2.0 * (x * x + y * y)
    return np.array([-gx, -gy, -gz], dtype=np.float32)


def _clip_command(command: Sequence[float], limits: np.ndarray) -> np.ndarray:
    cmd = np.asarray(command, dtype=np.float32).copy()
    for idx in range(3):
        cmd[idx] = np.clip(cmd[idx], limits[idx, 0], limits[idx, 1])
    return cmd


def _wait_for_key(valid_keys: set[str], prompt: str) -> str:
    valid = {key.lower() for key in valid_keys}
    print(prompt)
    if sys.stdin is None or not sys.stdin.isatty() or termios is None or tty is None:
        while True:
            value = input().strip().lower()
            if value in valid:
                return value
            print(f"[WARN] Invalid input '{value}'. Valid keys: {sorted(valid)}")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1).lower()
            if key in valid:
                print(key)
                return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class JitPolicy:
    def __init__(self, module: torch.jit.ScriptModule, device: torch.device):
        self.module = module.to(device)
        self.device = device

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            action = self.module(obs_tensor)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


class CheckpointPolicy:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            action = self.model(obs_tensor)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _build_actor_from_state_dict(
    state_dict: dict,
    num_obs: int,
    num_actions: int,
    activation: str,
) -> nn.Sequential:
    actor_weights: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.startswith("actor.") or not key.endswith(".weight"):
            continue
        layer_idx = int(key.split(".")[1])
        bias_key = f"actor.{layer_idx}.bias"
        if bias_key not in state_dict:
            raise RuntimeError(f"missing bias for {key}")
        actor_weights.append((layer_idx, value, state_dict[bias_key]))

    if not actor_weights:
        raise RuntimeError("checkpoint does not contain actor.* weights")

    actor_weights.sort(key=lambda item: item[0])
    act_ctor = _activation_factory(activation)
    layers: list[nn.Module] = []
    prev_out = num_obs

    for idx, (_layer_idx, weight, bias) in enumerate(actor_weights):
        out_dim, in_dim = weight.shape
        if in_dim != prev_out:
            raise RuntimeError(
                f"actor layer input mismatch at layer {idx}: expected {prev_out}, got {in_dim}"
            )
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        prev_out = out_dim
        if idx != len(actor_weights) - 1:
            layers.append(act_ctor())

    if prev_out != num_actions:
        raise RuntimeError(f"final actor output dim is {prev_out}, expected {num_actions}")

    return nn.Sequential(*layers)


def load_policy(policy_path: Path, num_obs: int, num_actions: int, activation: str, device: torch.device):
    try:
        module = torch.jit.load(str(policy_path), map_location=device)
        test_out = module(torch.zeros(1, num_obs, device=device))
        if tuple(test_out.shape) != (1, num_actions):
            raise RuntimeError(
                f"jit policy output shape {tuple(test_out.shape)} does not match {(1, num_actions)}"
            )
        print(f"[INFO] Loaded TorchScript policy: {policy_path}")
        return JitPolicy(module, device)
    except Exception as jit_exc:
        checkpoint = torch.load(str(policy_path), map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"unsupported policy file: {policy_path}") from jit_exc
        actor = _build_actor_from_state_dict(state_dict, num_obs, num_actions, activation)
        with torch.no_grad():
            test_out = actor(torch.zeros(1, num_obs))
        if tuple(test_out.shape) != (1, num_actions):
            raise RuntimeError(
                f"checkpoint actor output shape {tuple(test_out.shape)} does not match {(1, num_actions)}"
            ) from jit_exc
        print(f"[INFO] Loaded raw checkpoint actor: {policy_path}")
        return CheckpointPolicy(actor, device)


class WirelessRemoteState:
    def __init__(self) -> None:
        self.lx = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.ly = 0.0
        self.buttons = {name: False for name in REMOTE_BUTTONS}

    def parse(self, remote_data: Sequence[int]) -> bool:
        if remote_data is None or len(remote_data) < 24:
            return False
        data = bytes(int(v) & 0xFF for v in remote_data)
        self.lx = np.frombuffer(data[4:8], dtype=np.float32)[0]
        self.rx = np.frombuffer(data[8:12], dtype=np.float32)[0]
        self.ry = np.frombuffer(data[12:16], dtype=np.float32)[0]
        self.ly = np.frombuffer(data[20:24], dtype=np.float32)[0]

        data1 = data[2]
        data2 = data[3]
        self.buttons.update(
            {
                "r1": bool((data1 >> 0) & 1),
                "l1": bool((data1 >> 1) & 1),
                "start": bool((data1 >> 2) & 1),
                "select": bool((data1 >> 3) & 1),
                "r2": bool((data1 >> 4) & 1),
                "l2": bool((data1 >> 5) & 1),
                "f1": bool((data1 >> 6) & 1),
                "f3": bool((data1 >> 7) & 1),
                "a": bool((data2 >> 0) & 1),
                "b": bool((data2 >> 1) & 1),
                "x": bool((data2 >> 2) & 1),
                "y": bool((data2 >> 3) & 1),
                "up": bool((data2 >> 4) & 1),
                "right": bool((data2 >> 5) & 1),
                "down": bool((data2 >> 6) & 1),
                "left": bool((data2 >> 7) & 1),
            }
        )
        return True

    def pressed(self, button: str) -> bool:
        return bool(self.buttons.get(button.lower(), False))


class KeyboardTeleop:
    def __init__(self, runner: "Go2DeploySimToReal", step_sizes: np.ndarray):
        self.runner = runner
        self.step_sizes = np.asarray(step_sizes, dtype=np.float32)
        self._running = False
        self._old_tty_settings = None

    def start(self) -> None:
        if sys.stdin is None or not sys.stdin.isatty() or termios is None or tty is None:
            print("[WARN] stdin is not a tty; keyboard teleop is disabled.")
            return
        self._running = True
        fd = sys.stdin.fileno()
        self._old_tty_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print("[INFO] Keyboard teleop: w/s=vx, a/d=vy, q/e=wz, space=zero, x=stop")

    def stop(self) -> None:
        self._running = False
        if self._old_tty_settings is not None and sys.stdin is not None and termios is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_tty_settings)
            except Exception:
                pass
            self._old_tty_settings = None

    def poll_once(self) -> None:
        if not self._running or sys.stdin is None:
            return
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return
        key = sys.stdin.read(1).lower()
        self._handle_key(key)

    def _handle_key(self, key: str) -> None:
        if key == "w":
            self.runner.update_command_delta(self.step_sizes[0], 0.0, 0.0)
        elif key == "s":
            self.runner.update_command_delta(-self.step_sizes[0], 0.0, 0.0)
        elif key == "a":
            self.runner.update_command_delta(0.0, self.step_sizes[1], 0.0)
        elif key == "d":
            self.runner.update_command_delta(0.0, -self.step_sizes[1], 0.0)
        elif key == "q":
            self.runner.update_command_delta(0.0, 0.0, self.step_sizes[2])
        elif key == "e":
            self.runner.update_command_delta(0.0, 0.0, -self.step_sizes[2])
        elif key == " ":
            self.runner.set_command(0.0, 0.0, 0.0)
        elif key in {"x", "\x03"}:
            self.runner.request_stop()
            self._running = False
            return
        else:
            return
        self.runner.print_command(prefix="\rcommand")


class Go2DeploySimToReal:
    def __init__(self, args: argparse.Namespace):
        _load_unitree_sdk()
        self.args = args
        self.device = torch.device(args.device)
        self.num_obs = 47
        self.num_actions = 12
        self.control_dt = float(args.dt)
        self.gait_period = float(args.gait_period)
        self.cmd_scale = np.array(args.cmd_scale, dtype=np.float32)
        if args.no_command_limits:
            self.command_limits = np.array(
                [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]],
                dtype=np.float32,
            )
        else:
            self.command_limits = np.array(
                [
                    [args.vx_min, args.vx_max],
                    [args.vy_min, args.vy_max],
                    [args.wz_min, args.wz_max],
                ],
                dtype=np.float32,
            )
        self.kps = np.full(self.num_actions, float(args.kp), dtype=np.float32)
        self.kds = np.full(self.num_actions, float(args.kd), dtype=np.float32)
        self.default_joint_pos = DEFAULT_JOINT_POS.astype(np.float32).copy()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_history = [
            np.zeros(self.num_actions, dtype=np.float32)
            for _ in range(max(1, int(args.action_delay_steps) + 1))
        ]

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._debug_last_print = 0.0
        self._remote_last_print = 0.0

        self.command = _clip_command([args.vx, args.vy, args.wz], self.command_limits)
        self.remote_state = WirelessRemoteState()
        self.low_state: LowState_ | None = None

        self.policy = load_policy(
            policy_path=args.policy,
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            activation=args.activation,
            device=self.device,
        )

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self._init_lowcmd_template()
        self._init_communication()
        self._release_high_level_mode()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _init_lowcmd_template(self) -> None:
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = POS_STOP_F
            self.low_cmd.motor_cmd[i].dq = VEL_STOP_F
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

    def _init_communication(self) -> None:
        if self.args.channel is None:
            ChannelFactoryInitialize(int(self.args.domain))
        else:
            ChannelFactoryInitialize(int(self.args.domain), self.args.channel)

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowstate_handler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def _release_high_level_mode(self) -> None:
        while True:
            status, result = self.msc.CheckMode()
            if status != 0:
                print(f"[WARN] MotionSwitcher CheckMode failed: status={status}, result={result}")
                return

            mode_name = ""
            if isinstance(result, dict):
                mode_name = str(result.get("name", ""))
            elif hasattr(result, "name"):
                mode_name = str(getattr(result, "name"))

            if not mode_name:
                return

            print(f"[INFO] Releasing current high-level mode: {mode_name}")
            self.sc.StandDown()
            self.msc.ReleaseMode()
            time.sleep(1.0)

    def _lowstate_handler(self, msg: LowState_) -> None:
        with self._state_lock:
            self.low_state = msg
        if self.args.teleop in {"remote", "both"}:
            self._update_command_from_remote(msg.wireless_remote)

    def _get_low_state(self) -> LowState_ | None:
        with self._state_lock:
            return self.low_state

    def wait_for_low_state(self, timeout_sec: float = 10.0) -> bool:
        started = time.time()
        while True:
            if self._get_low_state() is not None:
                return True
            if timeout_sec > 0.0 and time.time() - started > timeout_sec:
                return False
            time.sleep(0.01)

    def get_command(self) -> np.ndarray:
        with self._command_lock:
            return self.command.copy()

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        with self._command_lock:
            self.command[:] = _clip_command([vx, vy, wz], self.command_limits)

    def update_command_delta(self, dvx: float, dvy: float, dwz: float) -> None:
        with self._command_lock:
            self.command[:] = _clip_command(
                self.command + np.array([dvx, dvy, dwz], dtype=np.float32),
                self.command_limits,
            )

    def print_command(self, prefix: str = "command") -> None:
        cmd = self.get_command()
        print(f"{prefix} -> vx:{cmd[0]:+.2f}, vy:{cmd[1]:+.2f}, wz:{cmd[2]:+.2f}", end="", flush=True)

    @staticmethod
    def _deadband_axis(value: float, deadband: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        deadband = abs(float(deadband))
        if abs(value) <= deadband:
            return 0.0
        return math.copysign((abs(value) - deadband) / max(1.0 - deadband, 1e-6), value)

    def _update_command_from_remote(self, remote_data: Sequence[int]) -> None:
        if not self.remote_state.parse(remote_data):
            return

        stop_button = str(self.args.remote_stop_button).lower()
        if stop_button != "none" and self.remote_state.pressed(stop_button):
            print("\n[INFO] Remote stop button pressed.")
            self.request_stop()
            return

        zero_button = str(self.args.remote_zero_button).lower()
        if zero_button != "none" and self.remote_state.pressed(zero_button):
            self.set_command(0.0, 0.0, 0.0)
            return

        deadman_button = str(self.args.remote_deadman_button).lower()
        if deadman_button != "none" and not self.remote_state.pressed(deadman_button):
            self.set_command(0.0, 0.0, 0.0)
            return

        lx = self._deadband_axis(self.remote_state.lx, self.args.remote_deadband)
        ly = self._deadband_axis(self.remote_state.ly, self.args.remote_deadband)
        rx = self._deadband_axis(self.remote_state.rx, self.args.remote_deadband)
        vx = ly * float(self.args.remote_vx_scale)
        vy = lx * float(self.args.remote_vy_scale)
        wz = rx * float(self.args.remote_wz_scale)
        self.set_command(vx, vy, wz)

        now = time.monotonic()
        if self.args.remote_debug and now - self._remote_last_print > float(self.args.debug_interval_s):
            self._remote_last_print = now
            cmd = self.get_command()
            print(
                "\n[REMOTE] "
                f"lx={self.remote_state.lx:+.2f}, ly={self.remote_state.ly:+.2f}, "
                f"rx={self.remote_state.rx:+.2f}, cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})"
            )

    def _get_sdk_motor_pos(self) -> np.ndarray:
        msg = self._get_low_state()
        pos = np.zeros(12, dtype=np.float32)
        if msg is None:
            return pos
        for idx in range(12):
            pos[idx] = float(msg.motor_state[idx].q)
        return pos

    def _publish_sdk_pose(self, target_pos_sdk: np.ndarray, kp: float, kd: float) -> None:
        for motor_idx in range(min(12, target_pos_sdk.size)):
            cmd = self.low_cmd.motor_cmd[motor_idx]
            cmd.mode = 0x01
            cmd.q = float(target_pos_sdk[motor_idx])
            cmd.dq = 0.0
            cmd.kp = float(kp)
            cmd.kd = float(kd)
            cmd.tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _interpolate_pose(self, start_pos_sdk: np.ndarray, end_pos_sdk: np.ndarray, duration_sec: float) -> None:
        steps = max(1, int(max(0.0, float(duration_sec)) / 0.002))
        for step_idx in range(1, steps + 1):
            if self.stop_requested:
                return
            ratio = float(step_idx) / float(steps)
            pose = (1.0 - ratio) * start_pos_sdk + ratio * end_pos_sdk
            self._publish_sdk_pose(pose, kp=self.args.startup_kp, kd=self.args.startup_kd)
            time.sleep(0.002)

    def _hold_pose(self, target_pos_sdk: np.ndarray, hold_sec: float, kp: float, kd: float) -> None:
        steps = max(1, int(max(0.0, float(hold_sec)) / 0.002))
        for _ in range(steps):
            if self.stop_requested:
                return
            self._publish_sdk_pose(target_pos_sdk, kp=kp, kd=kd)
            time.sleep(0.002)

    def _hold_pose_until_key(
        self,
        target_pos_sdk: np.ndarray,
        valid_keys: set[str],
        prompt: str,
        kp: float,
        kd: float,
    ) -> str:
        valid = {key.lower() for key in valid_keys}
        print(prompt)

        if sys.stdin is None or not sys.stdin.isatty() or termios is None or tty is None:
            while not self.stop_requested:
                self._publish_sdk_pose(target_pos_sdk, kp=kp, kd=kd)
                readable, _, _ = select.select([sys.stdin], [], [], 0.002)
                if not readable:
                    continue
                value = sys.stdin.readline().strip().lower()
                if value in valid:
                    return value
                print(f"[WARN] Invalid input '{value}'. Valid keys: {sorted(valid)}")
            return ""

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not self.stop_requested:
                self._publish_sdk_pose(target_pos_sdk, kp=kp, kd=kd)
                readable, _, _ = select.select([sys.stdin], [], [], 0.002)
                if not readable:
                    continue
                key = sys.stdin.read(1).lower()
                if key in valid:
                    print(key)
                    return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ""

    def run_startup_sequence(self) -> None:
        if not self.wait_for_low_state():
            raise RuntimeError("LowState not received. Cannot start real deployment.")

        self.set_command(0.0, 0.0, 0.0)
        _wait_for_key({"1"}, "[INFO] Press '1' to move current pose -> sit pose.")
        self._interpolate_pose(
            self._get_sdk_motor_pos(),
            DEFAULT_STARTUP_SIT_POSE_SDK,
            self.args.startup_transition_sec,
        )
        self._hold_pose(DEFAULT_STARTUP_SIT_POSE_SDK, 0.25, self.args.startup_kp, self.args.startup_kd)
        print("[INFO] Sit pose reached.")

        _wait_for_key({"2"}, "[INFO] Press '2' to move sit pose -> stand pose.")
        self._interpolate_pose(
            self._get_sdk_motor_pos(),
            DEFAULT_STARTUP_STAND_POSE_SDK,
            self.args.startup_transition_sec,
        )
        self._hold_pose(
            DEFAULT_STARTUP_STAND_POSE_SDK,
            self.args.startup_hold_sec,
            self.args.startup_kp,
            self.args.startup_kd,
        )
        if not self.args.auto_start:
            self._hold_pose_until_key(
                DEFAULT_STARTUP_STAND_POSE_SDK,
                {"3"},
                "[INFO] Holding stand pose. Press '3' to hand over to RL policy.",
                self.args.startup_kp,
                self.args.startup_kd,
            )

    def _build_observation(self, msg: LowState_, phase: float) -> np.ndarray:
        joint_pos = np.empty(self.num_actions, dtype=np.float32)
        joint_vel = np.empty(self.num_actions, dtype=np.float32)
        for policy_idx, motor_idx in enumerate(POLICY_TO_SDK_INDEX):
            state = msg.motor_state[int(motor_idx)]
            joint_pos[policy_idx] = float(state.q)
            joint_vel[policy_idx] = float(state.dq)

        base_ang_vel = np.array([float(v) for v in msg.imu_state.gyroscope], dtype=np.float32)
        projected_gravity = _quat_to_gravity_body(msg.imu_state.quaternion, self.args.quat_order)
        command = self.get_command()

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = base_ang_vel * self.args.ang_vel_scale
        obs[3:6] = projected_gravity
        obs[6:9] = command * self.cmd_scale
        obs[9:21] = (joint_pos - self.default_joint_pos) * self.args.dof_pos_scale
        obs[21:33] = joint_vel * self.args.dof_vel_scale
        obs[33:45] = self.last_action
        sin_phase = math.sin(2.0 * math.pi * phase)
        cos_phase = math.cos(2.0 * math.pi * phase)
        command_active = (
            np.linalg.norm(command[:2]) >= self.args.phase_command_threshold
            or abs(command[2]) >= self.args.phase_command_threshold
        )
        if self.args.stand_phase_lock and not command_active:
            sin_phase = 0.0
            cos_phase = 1.0
        obs[45] = sin_phase
        obs[46] = cos_phase
        return obs.reshape(1, -1)

    def _publish_policy_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.clip(raw_action.astype(np.float32), -self.args.clip_actions, self.args.clip_actions)
        self.action_history.append(action.copy())
        applied_action = self.action_history.pop(0).astype(np.float32, copy=False)

        target_pos_policy = self.default_joint_pos + applied_action * self.args.action_scale
        target_pos_policy = np.clip(target_pos_policy, self.args.q_min, self.args.q_max)

        for policy_idx, motor_idx in enumerate(POLICY_TO_SDK_INDEX):
            cmd = self.low_cmd.motor_cmd[int(motor_idx)]
            cmd.mode = 0x01
            cmd.q = float(target_pos_policy[policy_idx])
            cmd.dq = 0.0
            cmd.kp = float(self.kps[policy_idx])
            cmd.kd = float(self.kds[policy_idx])
            cmd.tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
        self.last_action = action
        return applied_action

    def run_policy_loop(self, keyboard: KeyboardTeleop | None) -> None:
        step_idx = 0
        next_tick = time.monotonic()
        print(f"[INFO] Starting RL loop at {1.0 / self.control_dt:.1f} Hz")
        print(
            "[INFO] policy order -> "
            + ", ".join(POLICY_JOINT_NAMES)
        )
        print(
            "[INFO] sdk order -> "
            + ", ".join(SDK_JOINT_NAMES)
        )

        while not self.stop_requested:
            if keyboard is not None:
                keyboard.poll_once()

            msg = self._get_low_state()
            if msg is None:
                time.sleep(0.002)
                continue

            phase_time = (step_idx * self.control_dt) % self.gait_period
            phase = phase_time / self.gait_period
            obs = self._build_observation(msg, phase)
            raw_action = self.policy(obs)
            applied_action = self._publish_policy_action(raw_action)

            now = time.monotonic()
            if self.args.debug and now - self._debug_last_print > float(self.args.debug_interval_s):
                self._debug_last_print = now
                cmd = self.get_command()
                print(
                    "\n[DEBUG] "
                    f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                    f"action=[{applied_action.min():+.3f},{applied_action.max():+.3f}] "
                    f"phase={phase:.3f}"
                )

            step_idx += 1
            next_tick += self.control_dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def close(self) -> None:
        try:
            current_sdk = self._get_sdk_motor_pos()
            self._hold_pose(current_sdk, hold_sec=0.1, kp=self.args.stop_kp, kd=self.args.stop_kd)
        except Exception:
            pass
        try:
            self.lowstate_subscriber.Close()
        except Exception:
            pass
        try:
            self.lowcmd_publisher.Close()
        except Exception:
            pass


def _get_latest_exported_policy(experiment_name: str) -> Path | None:
    experiment_dir = REPO_ROOT / "logs" / "rsl_rl" / experiment_name
    candidates = sorted(
        experiment_dir.glob("*/exported/policy.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def resolve_policy_path(args: argparse.Namespace) -> Path:
    if args.policy is not None:
        return Path(args.policy).expanduser().resolve()
    if args.load_run is not None:
        run_path = Path(args.load_run).expanduser()
        if not run_path.is_absolute():
            run_path = REPO_ROOT / "logs" / "rsl_rl" / args.experiment_name / run_path
        checkpoint_name = f"model_{int(args.checkpoint)}.pt"
        return (run_path / checkpoint_name).resolve()
    latest_policy = _get_latest_exported_policy(args.experiment_name)
    if latest_policy is not None:
        return latest_policy
    return DEFAULT_POLICY.resolve()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Go2 low-level sim-to-real runner for Isaac Lab policies.")
    parser.add_argument("channel", nargs="?", default=None, help="DDS network interface, e.g. enp3s0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--policy", type=str, default=None, help="Path to exported TorchScript policy.pt or raw model_XXXX.pt")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--load-run", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--activation", default="elu", choices=("elu", "relu", "selu", "lrelu", "tanh", "sigmoid"))

    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--gait-period", type=float, default=0.6)
    parser.set_defaults(stand_phase_lock=True)
    parser.add_argument(
        "--stand-phase-lock",
        dest="stand_phase_lock",
        action="store_true",
        help="freeze gait phase observation at zero command (default: enabled)",
    )
    parser.add_argument(
        "--no-stand-phase-lock",
        dest="stand_phase_lock",
        action="store_false",
        help="disable stand phase lock for ablation/debugging",
    )
    parser.add_argument("--phase-command-threshold", type=float, default=0.1)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--clip-actions", type=float, default=100.0)
    parser.add_argument("--action-delay-steps", type=int, default=0)
    parser.add_argument("--q-min", type=float, default=-3.2)
    parser.add_argument("--q-max", type=float, default=3.2)

    parser.add_argument("--ang-vel-scale", type=float, default=0.25)
    parser.add_argument("--dof-pos-scale", type=float, default=1.0)
    parser.add_argument("--dof-vel-scale", type=float, default=0.05)
    parser.add_argument("--cmd-scale", type=float, nargs=3, default=[2.0, 2.0, 0.25])

    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--vx-min", type=float, default=-0.5)
    parser.add_argument("--vx-max", type=float, default=1.0)
    parser.add_argument("--vy-min", type=float, default=-0.5)
    parser.add_argument("--vy-max", type=float, default=0.5)
    parser.add_argument("--wz-min", type=float, default=-1.0)
    parser.add_argument("--wz-max", type=float, default=1.0)
    parser.add_argument("--no-command-limits", action="store_true")
    parser.add_argument("--step-x", type=float, default=0.05)
    parser.add_argument("--step-y", type=float, default=0.05)
    parser.add_argument("--step-z", type=float, default=0.10)

    parser.add_argument("--teleop", choices=("keyboard", "remote", "both", "none"), default="keyboard")
    parser.add_argument(
        "--remote-deadman-button",
        choices=(*REMOTE_BUTTONS, "none"),
        default="r1",
    )
    parser.add_argument(
        "--remote-zero-button",
        choices=(*REMOTE_BUTTONS, "none"),
        default="b",
    )
    parser.add_argument(
        "--remote-stop-button",
        choices=(*REMOTE_BUTTONS, "none"),
        default="x",
    )
    parser.add_argument("--remote-deadband", type=float, default=0.08)
    parser.add_argument("--remote-vx-scale", type=float, default=1.0)
    parser.add_argument("--remote-vy-scale", type=float, default=-0.5)
    parser.add_argument("--remote-wz-scale", type=float, default=-1.0)
    parser.add_argument("--remote-debug", action="store_true")

    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--skip-startup-sequence", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--startup-transition-sec", type=float, default=1.5)
    parser.add_argument("--startup-hold-sec", type=float, default=1.0)
    parser.add_argument("--startup-kp", type=float, default=60.0)
    parser.add_argument("--startup-kd", type=float, default=5.0)
    parser.add_argument("--stop-kp", type=float, default=20.0)
    parser.add_argument("--stop-kd", type=float, default=2.0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-interval-s", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive safety confirmation prompt.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.policy = resolve_policy_path(args)

    print("WARNING: low-level control can move the robot abruptly.")
    print("Keep the robot on a flat floor, clear the area, and keep an estop ready.")
    print(f"[INFO] Using policy source: {args.policy}")
    if args.yes:
        print("[INFO] Skipping safety prompt because --yes was provided.")
    elif sys.stdin is None or not sys.stdin.isatty():
        if args.check:
            print("[INFO] No interactive stdin detected; continuing automatically for --check.")
        else:
            raise RuntimeError(
                "Interactive safety prompt is unavailable in this shell. "
                "Re-run with --yes only if you are physically ready to control the robot."
            )
    else:
        input("Press Enter to initialize DDS and load the policy...")

    runner = Go2DeploySimToReal(args)
    if args.check:
        print("[INFO] Check OK. Policy load and DDS initialization succeeded.")
        runner.close()
        return

    keyboard = None
    try:
        if not args.skip_startup_sequence:
            runner.run_startup_sequence()
        else:
            if not runner.wait_for_low_state():
                raise RuntimeError("LowState not received.")
            if not args.auto_start:
                _wait_for_key({"3"}, "[INFO] Startup skipped. Press '3' to start RL.")

        if args.teleop in {"keyboard", "both"}:
            keyboard = KeyboardTeleop(
                runner,
                step_sizes=np.array([args.step_x, args.step_y, args.step_z], dtype=np.float32),
            )
            keyboard.start()
        elif args.teleop == "remote":
            print("[INFO] Remote teleop enabled. Hold R1 by default for nonzero command.")

        runner.run_policy_loop(keyboard)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Stopping.")
        runner.request_stop()
    finally:
        if keyboard is not None:
            keyboard.stop()
        runner.close()
        print("\n[INFO] go2_sim_to_real.py exited.")


if __name__ == "__main__":
    main()

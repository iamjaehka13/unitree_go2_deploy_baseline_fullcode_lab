# Go2 Sim-To-Real

This folder contains the Go2 low-level sim-to-real runner used by this baseline.
For the Isaac Lab version, pass an exported TorchScript policy explicitly.

## Usage

```bash
export UNITREE_SDK2_PYTHON_ROOT=/path/to/unitree_sdk2_python
python deploy/deploy_real/go2_sim_to_real.py enp6s0 --policy /path/to/exported/policy.pt
```

## Safety

Low-level control can move the robot abruptly. Confirm the robot is ready, clear the area, and keep an emergency stop available before running.

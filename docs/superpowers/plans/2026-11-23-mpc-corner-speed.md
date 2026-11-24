# MPC corner speed profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The MPC's horizon reference slows to `v_turn` at corners and changes speed at most at `a_ref`.

**Architecture:** `MpcPathController` builds a speed profile over arc length whenever the path changes; `_speed(s)` reads it. Nothing else changes.

**Tech Stack:** Python, numpy, pytest (in the `maze-sim` distrobox).

**Spec:** `docs/superpowers/specs/2026-11-23-mpc-corner-speed-design.md`

## Global Constraints

- Off when `v_turn >= v_ref`: the reference is exactly the current constant-`v_ref` one.
- New keys in `config/mpc.yaml` `mpc:`: `v_turn`, `a_ref`, `corner_window` (floats, written with a decimal point).
- Commit titles start with `[control]`; no Claude attribution.

## Review Focus

- A path shorter than `corner_window`, a single-point path, an empty path: no NaN, no crash.
- A U-turn (two 90° corners 15 cm apart): the speed between them never exceeds what `a_ref` allows from both.
- A path republished unchanged keeps the profile (no rebuild needed, no reset).
- `a_ref` very large: profile equals the caps; `a_ref` tiny: profile ≈ `v_turn` everywhere near corners.
- A staircase path: no false slowdown below ~0.85 `v_ref`.

---

### Task 1: Speed profile

**Files:**
- Modify: `src/maze_control/maze_control/mpc/controller.py`
- Modify: `src/maze_control/config/mpc.yaml`
- Test: `src/maze_control/test/test_mpc.py`

**Interfaces:**
- Produces: `MpcPathController(tracker, law, v_ref, slow_dist, v_turn=None, a_ref=inf, corner_window=0.02, **kw)`; `ctrl.speed_profile` → `(s_grid (n,), v (n,))` or `None`; `_speed(s)` unchanged signature.

- [ ] **Step 1: Failing tests** in `test_mpc.py`: `test_profile_off_is_constant_v_ref`, `test_profile_straight_runs_at_v_ref`, `test_profile_slows_to_v_turn_at_corner` (L: speed at s = 0.30 is `v_turn`; ≤ √(v_turn² + 2 a_ref |s − 0.30|) around it; `v_ref` 0.15 m before), `test_profile_u_turn_bounded_by_both_corners`, `test_profile_staircase_is_not_a_corner` (min ≥ 0.85 v_ref), `test_horizon_velocity_follows_profile`, `test_profile_degenerate_paths`.
- [ ] **Step 2: Run** `pytest test/test_mpc.py -k profile` → FAIL (no `speed_profile`).
- [ ] **Step 3: Implement** the profile per the spec; `from_params` reads the keys with off defaults.
- [ ] **Step 4: Run** `pytest test/test_mpc.py` → PASS (old tests unchanged).
- [ ] **Step 5: Commit** `[control] MPC: reference slows before corners (v_turn, a_ref)`.

### Task 2: Defaults from a sweep

**Files:**
- Modify: `src/maze_control/config/mpc.yaml`, `README.md`, `src/maze_control/scripts/compare_controllers.py` (uncommitted, add an "MPC corner" variant)

- [ ] **Step 1:** Sweep `v_ref ∈ {0.15, 0.2}`, `v_turn ∈ {0.05, 0.07, 0.09}`, `a_ref ∈ {0.15, 0.3}` on L and U, 20 randomized runs each, ±5 mm corridors; record time and in-corridor rate.
- [ ] **Step 2:** Pick defaults: in-corridor rate ≥ the `v_ref = 0.1` MPC's, fastest time. Write them into `mpc.yaml`; README MPC paragraph updated.
- [ ] **Step 3: Run** the full package tests → PASS.
- [ ] **Step 4: Commit** `[control] MPC: corner-speed defaults from the offline sweep`.

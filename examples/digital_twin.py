#!/usr/bin/env python3
"""A two-joint digital body for a hip exoskeleton, driven by identified parameters.

Run it with::

    uv run python examples/digital_twin.py

Why this exists
---------------
Downstream wants to pull the USB cable out mid-demo and keep going.  That only works if
something with the *same interface as the real bridge* can take over: same method names,
same sample record, same units.  ``DigitalTwin`` below is that stand-in.  It opens no
serial port, imports nothing hardware-related and never reads a clock -- the caller owns
time, so a replay is bit-identical every run.

Division of labour
------------------
``sim2real_actuator`` models the **actuator** (Coulomb + viscous friction, power-on bias,
backlash, torque limit).  Everything else -- the limb, the straps, the end stops -- is the
**plant**, and lives here in the example as ``JointLoad``, fed back through
``ActuatorSim.step(external_nm=...)``.  Keeping the plant out of the library is what lets
the same library drive a duck's leg, a gripper or a hip exo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from sim2real_actuator import (
    ActuatorParams,
    ActuatorSim,
    DomainRanges,
    SimState,
    identify,
    load_columns,
    randomize,
)

DATA = Path(__file__).parent / "data"
RAMP = DATA / "breakaway-measurement.csv"
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

# --- assumptions, stated out loud -------------------------------------------------------
# The recorded velocity column is quantised to 0.1 deg/s at ~5 ms, so differentiating it is
# mostly noise and identify() cannot recover the inertia (it is asked for the friction
# model only, with fit_inertia=False).  This value is an order-of-magnitude estimate read
# off the break-away ramp's rise time (~0.45 rad/s reached in ~0.3 s under ~0.1 Nm of net
# torque), NOT a least-squares fit.  Replace it with a real number when you have one.
ASSUMED_INERTIA_KG_M2 = 0.05

# The plant: a soft return spring plus strap damping, again an assumption of this example.
PLANT_SPRING_NM_PER_RAD = 0.5
PLANT_DAMPING_NM_S_PER_RAD = 0.05
PLANT_STOP_DEG = 80.0
PLANT_STOP_STIFFNESS_NM_PER_RAD = 40.0


@dataclass(frozen=True)
class TwinSample:
    """Same field names and units as the real bridge's ``Sample``."""

    host_t: float
    ms: float
    pitch: float
    roll: float
    yaw: float
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float
    kpa: float
    ldeg: float
    rdeg: float
    ldps: float
    rdps: float
    cmd_l: float = 0.0
    cmd_r: float = 0.0


class JointLoad:
    """Everything acting on the joint that is not the actuator.

    A soft return spring towards ``rest_rad``, strap damping, and stiff end stops.
    Stateless and pure: it only looks at the current position and velocity.
    """

    __slots__ = ("damping", "rest_rad", "spring", "stop_rad", "stop_stiffness")

    def __init__(
        self,
        rest_rad: float,
        *,
        spring: float = PLANT_SPRING_NM_PER_RAD,
        damping: float = PLANT_DAMPING_NM_S_PER_RAD,
        stop_rad: float = PLANT_STOP_DEG * DEG2RAD,
        stop_stiffness: float = PLANT_STOP_STIFFNESS_NM_PER_RAD,
    ) -> None:
        self.rest_rad = rest_rad
        self.spring = spring
        self.damping = damping
        self.stop_rad = stop_rad
        self.stop_stiffness = stop_stiffness

    def torque_nm(self, pos_rad: float, vel_rad_s: float) -> float:
        torque = -self.spring * (pos_rad - self.rest_rad) - self.damping * vel_rad_s
        overrun = abs(pos_rad - self.rest_rad) - self.stop_rad
        if overrun > 0.0:
            direction = 1.0 if pos_rad > self.rest_rad else -1.0
            torque -= direction * self.stop_stiffness * overrun
        return torque


class DigitalTwin:
    """Stand-in for the real hip exoskeleton: two actuators, no hardware, no clock.

    Mirrors the real bridge's surface (``open`` / ``ping`` / ``version`` / ``on_sample`` /
    ``set_torque`` / ``commanded`` / ``stream_hz`` / ``enable`` / ``disable`` / ``close``).
    The one difference is deliberate: the real bridge streams from its own thread, while
    the twin advances only when the caller calls :meth:`advance`, so replays are exactly
    reproducible.
    """

    def __init__(
        self,
        left: ActuatorParams,
        right: ActuatorParams,
        *,
        start_ldeg: float,
        start_rdeg: float,
        t0: float = 0.0,
        nominal_hz: float = 200.0,
    ) -> None:
        self._sims = {
            "l": ActuatorSim(left, seed=0),
            "r": ActuatorSim(right, seed=1),
        }
        self._loads = {
            "l": JointLoad(start_ldeg * DEG2RAD),
            "r": JointLoad(start_rdeg * DEG2RAD),
        }
        self._sims["l"].reset(start_ldeg * DEG2RAD, 0.0)
        self._sims["r"].reset(start_rdeg * DEG2RAD, 0.0)
        self._t0 = t0
        self._t = t0
        self._target = (0.0, 0.0)
        self._enabled = False
        self._send_torque = False
        self._nominal_hz = nominal_hz
        self._callbacks: list[Callable[[TwinSample], None]] = []
        self._last = self._make_sample(
            SimState(start_ldeg * DEG2RAD, 0.0, 0.0, False),
            SimState(start_rdeg * DEG2RAD, 0.0, 0.0, False),
        )

    # -- the real bridge's surface --------------------------------------------------------
    def open(self) -> DigitalTwin:
        return self

    def ping(self) -> str:
        return "PONG(twin)"

    def version(self) -> str:
        return "OK,VERSION,digital-twin"

    def on_sample(self, cb: Callable[[TwinSample], None]) -> None:
        self._callbacks.append(cb)

    def set_torque(self, left_nm: float, right_nm: float) -> None:
        self._target = (float(left_nm), float(right_nm))

    def commanded(self) -> tuple[float, float]:
        return self._target

    def stream_hz(self) -> float:
        return self._nominal_hz

    def enable(self, send_torque: bool = False) -> None:
        self._enabled = True
        self._send_torque = send_torque

    def disable(self) -> None:
        self._enabled = False
        self._send_torque = False
        self._target = (0.0, 0.0)

    def close(self) -> None:
        self.disable()

    # -- the twin's own knob: the caller owns time ----------------------------------------
    def advance(self, dt_s: float) -> TwinSample:
        """Integrate both joints by ``dt_s`` and emit one sample."""
        cmd_l, cmd_r = self._target if (self._enabled and self._send_torque) else (0.0, 0.0)
        states: dict[str, SimState] = {}
        for key, cmd in (("l", cmd_l), ("r", cmd_r)):
            sim, load = self._sims[key], self._loads[key]
            previous = self._last
            pos = (previous.ldeg if key == "l" else previous.rdeg) * DEG2RAD
            vel = (previous.ldps if key == "l" else previous.rdps) * DEG2RAD
            states[key] = sim.step(cmd, dt_s, external_nm=load.torque_nm(pos, vel))
        self._t += dt_s
        sample = self._make_sample(states["l"], states["r"], cmd_l=cmd_l, cmd_r=cmd_r)
        self._last = sample
        for cb in self._callbacks:
            cb(sample)
        return sample

    def _make_sample(
        self, left: SimState, right: SimState, *, cmd_l: float = 0.0, cmd_r: float = 0.0
    ) -> TwinSample:
        return TwinSample(
            host_t=self._t,
            ms=(self._t - self._t0) * 1000.0,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            gx=0.0,
            gy=0.0,
            gz=0.0,
            ax=0.0,
            ay=0.0,
            az=1.0,
            kpa=101.325,
            ldeg=left.pos_rad * RAD2DEG,
            rdeg=right.pos_rad * RAD2DEG,
            ldps=left.vel_rad_s * RAD2DEG,
            rdps=right.vel_rad_s * RAD2DEG,
            cmd_l=cmd_l,
            cmd_r=cmd_r,
        )


def identified_params() -> tuple[ActuatorParams, ActuatorParams, dict[str, list[float]]]:
    """Identify both joints off the real ramp and attach the assumed inertia."""
    out: list[ActuatorParams] = []
    real: dict[str, list[float]] = {}
    for tau, vel, pos in (("cmd_l", "ldps", "ldeg"), ("cmd_r", "rdps", "rdeg")):
        data = load_columns(str(RAMP), t="host_t", tau=tau, vel=vel, pos=pos, vel_unit="deg/s")
        params = identify(**data, fit_inertia=False)
        out.append(
            replace(
                params,
                inertia_kg_m2=ASSUMED_INERTIA_KG_M2,
                tau_limit_nm=1.5,
                source=f"{params.source}|inertia assumed {ASSUMED_INERTIA_KG_M2} kg*m^2",
            )
        )
        real[tau] = data["tau_nm"]
        real[vel] = data["vel_rad_s"]
        real[pos] = data["pos_rad"]
        real["t"] = data["t_s"]
    return out[0], out[1], real


def _span(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def demo_torque_sequence(left: ActuatorParams, right: ActuatorParams) -> None:
    """The literal acceptance test: a torque sequence in, angle/velocity trajectory out."""
    print("=" * 92)
    print("A. torque sequence -> angle / angular-velocity trajectory")
    print("=" * 92)
    twin = DigitalTwin(left, right, start_ldeg=-36.0, start_rdeg=34.0).open()
    twin.enable(send_torque=True)
    dt = 0.005
    print(
        f"{'t_s':>6s} {'cmd_l':>7s} {'cmd_r':>7s} {'ldeg':>8s} {'rdeg':>8s} "
        f"{'ldps':>8s} {'rdps':>8s}"
    )
    print("-" * 92)
    for step in range(600):
        t = step * dt
        drive = 0.6 * math.sin(2 * math.pi * 0.5 * t)
        twin.set_torque(drive, -drive)
        sample = twin.advance(dt)
        if step % 60 == 0:
            print(
                f"{t:6.2f} {sample.cmd_l:7.3f} {sample.cmd_r:7.3f} {sample.ldeg:8.2f} "
                f"{sample.rdeg:8.2f} {sample.ldps:8.2f} {sample.rdps:8.2f}"
            )
    twin.close()
    print()


def demo_replay_against_real(
    left: ActuatorParams, right: ActuatorParams, real: dict[str, list[float]]
) -> bool:
    """Push the *recorded* torque commands through the twin and compare the scale."""
    print("=" * 92)
    print("B. replay the recorded torque commands through the twin, compare with the real log")
    print("=" * 92)
    t = real["t"]
    twin = DigitalTwin(
        left,
        right,
        start_ldeg=real["ldeg"][0] * RAD2DEG,
        start_rdeg=real["rdeg"][0] * RAD2DEG,
        t0=t[0],
    ).open()
    twin.enable(send_torque=True)

    sim_ldeg: list[float] = []
    sim_rdeg: list[float] = []
    sim_ldps: list[float] = []
    sim_rdps: list[float] = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0.0:
            continue
        twin.set_torque(real["cmd_l"][i], real["cmd_r"][i])
        sample = twin.advance(dt)
        sim_ldeg.append(sample.ldeg)
        sim_rdeg.append(sample.rdeg)
        sim_ldps.append(sample.ldps)
        sim_rdps.append(sample.rdps)
    twin.close()

    rows = [
        ("left  angle span, deg", _span([p * RAD2DEG for p in real["ldeg"]]), _span(sim_ldeg)),
        ("right angle span, deg", _span([p * RAD2DEG for p in real["rdeg"]]), _span(sim_rdeg)),
        (
            "left  peak speed, deg/s",
            max(abs(v) for v in real["ldps"]) * RAD2DEG,
            max(abs(v) for v in sim_ldps),
        ),
        (
            "right peak speed, deg/s",
            max(abs(v) for v in real["rdps"]) * RAD2DEG,
            max(abs(v) for v in sim_rdps),
        ),
    ]
    print(f"{'quantity':28s} {'recorded':>12s} {'digital twin':>14s} {'ratio':>8s}  verdict")
    print("-" * 92)
    ok = True
    for label, measured, simulated in rows:
        ratio = simulated / measured if measured else float("inf")
        same_scale = 0.1 <= ratio <= 10.0
        ok &= same_scale
        print(
            f"{label:28s} {measured:12.2f} {simulated:14.2f} {ratio:8.2f}  "
            f"{'same order of magnitude' if same_scale else 'OFF BY MORE THAN 10x'}"
        )
    print()
    return ok


def demo_domain_randomization(left: ActuatorParams, right: ActuatorParams) -> None:
    """Twenty slightly different bodies, same torque sequence -- how wide is the spread?"""
    print("=" * 92)
    print("C. domain randomization: 20 bodies drawn from one identification")
    print("=" * 92)
    ranges = DomainRanges()
    spans: list[float] = []
    for l_p, r_p in zip(
        randomize(left, ranges, 20, seed=2024), randomize(right, ranges, 20, seed=4048), strict=True
    ):
        twin = DigitalTwin(l_p, r_p, start_ldeg=-36.0, start_rdeg=34.0).open()
        twin.enable(send_torque=True)
        angles: list[float] = []
        for step in range(400):
            t = step * 0.005
            twin.set_torque(0.6 * math.sin(2 * math.pi * 0.5 * t), 0.0)
            angles.append(twin.advance(0.005).ldeg)
        twin.close()
        spans.append(_span(angles))
    spans.sort()
    print(
        f"left-joint angle span across 20 bodies, deg: min={spans[0]:.2f} "
        f"median={spans[len(spans) // 2]:.2f} max={spans[-1]:.2f}"
    )
    print("Same seeds -> same twenty bodies -> same spread, on any machine.")
    print()


def main() -> int:
    left, right, real = identified_params()
    print("identified from", RAMP.name)
    print("  left :", left.to_dict())
    print("  right:", right.to_dict())
    print()
    demo_torque_sequence(left, right)
    ok = demo_replay_against_real(left, right, real)
    demo_domain_randomization(left, right)
    if ok:
        print(
            "ACCEPTED: the digital twin's trajectories are the same order of magnitude "
            "as the real recording."
        )
        return 0
    print("REJECTED: the digital twin is off by more than 10x somewhere.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

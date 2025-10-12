"""Defines the base actuators class, along with some implementations."""

__all__ = [
    "Actuators",
    "StatefulActuators",
    "TorqueActuators",
    "PositionActuators",
    "PositionVelocityActuator",
]

import logging
from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.noise import Noise, NoNoise
from ksim.types import Metadata, PhysicsData, PhysicsModel
from ksim.utils.mujoco import get_ctrl_data_idx_by_name

logger = logging.getLogger(__name__)


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def get_ctrl(self, action: Array, physics_data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the action vector."""

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        """Get the default action for the actuators."""
        return physics_data.ctrl


class StatefulActuators(Actuators):
    @abstractmethod
    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Get the control signal from the action vector."""

    @abstractmethod
    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PyTree:
        """Get the initial state for the actuator."""

    def get_ctrl(self, action: Array, physics_data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        raise NotImplementedError("Stateful actuators should use `get_stateful_ctrl` instead.")


class TorqueActuators(Actuators):
    """Direct torque control."""

    def __init__(self, noise: Noise | None = None) -> None:
        super().__init__()

        self.noise = NoNoise() if noise is None else noise

    def get_ctrl(self, action: Array, physics_data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Just use the action as the torque, the simplest actuator model."""
        return self.noise.add_noise(action, curriculum_level, rng)


class _PositionActuatorsBase:
    """Base initialization for position actuators - builds kp, kd, limits from metadata."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        action_noise: Noise | None = None,
        torque_noise: Noise | None = None,
        action_scale: float = 1.0,
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        ctrl_name_to_idx = get_ctrl_data_idx_by_name(physics_model)
        kps_list = [-1.0] * len(ctrl_name_to_idx)
        kds_list = [-1.0] * len(ctrl_name_to_idx)
        ctrl_clip_list = [jnp.inf] * len(ctrl_name_to_idx)

        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is required for MITPositionActuators")
        joint_name_to_metadata = metadata.joint_name_to_metadata

        for joint_name, params in joint_name_to_metadata.items():
            actuator_name = self.get_actuator_name(joint_name)
            if actuator_name not in ctrl_name_to_idx:
                if actuator_name != "root":
                    logger.warning("Joint %s has no actuator name. Skipping.", joint_name)
                continue
            actuator_idx = ctrl_name_to_idx[actuator_name]

            kp = params.kp
            kd = params.kd
            ctrl_clip = params.soft_torque_limit
            assert kp is not None and kd is not None, f"Missing kp or kd for joint {joint_name}"

            if ctrl_clip is not None:
                if ctrl_clip < 0:
                    raise ValueError(f"Soft torque limit for joint {joint_name} is negative: {ctrl_clip}")
                ctrl_clip_list[actuator_idx] = ctrl_clip

            kps_list[actuator_idx] = kp
            kds_list[actuator_idx] = kd

        if any(kp == -1 for kp in kps_list):
            raise ValueError("Some KPs are not set. Check the provided metadata.")
        if any(kd == -1 for kd in kds_list):
            raise ValueError("Some KDs are not set. Check the provided metadata.")

        self.kps = jnp.array(kps_list)
        self.kds = jnp.array(kds_list)
        self.ctrl_clip = jnp.array(ctrl_clip_list)

        self.action_noise = NoNoise() if action_noise is None else action_noise
        self.torque_noise = NoNoise() if torque_noise is None else torque_noise

        self.action_scale = action_scale

        if any(self.kps < 0) or any(self.kds < 0):
            raise ValueError("Some KPs or KDs are negative. Check the provided metadata.")
        if any(self.kps == 0) or any(self.kds == 0):
            logger.warning("Some KPs or KDs are 0. Check the provided metadata.")

    def get_actuator_name(self, joint_name: str) -> str:
        # This can be overridden if necessary.
        return f"{joint_name}_ctrl"


class PositionActuators(_PositionActuatorsBase, StatefulActuators):
    """MIT Cheetah-style actuator controller with per-episode randomized gains, action bias, and torque bias."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        action_noise: Noise | None = None,
        torque_noise: Noise | None = None,
        action_scale: float = 1.0,
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        action_bias_scale: float = 0.0,
        torque_bias_scale: float = 0.0,
    ) -> None:
        # Reuse base initialization to build base gains/limits and noises.
        _PositionActuatorsBase.__init__(self, physics_model, metadata, action_noise, torque_noise, action_scale)

        if kp_scale <= 0:
            raise ValueError("kp_scale must be positive")
        if kd_scale <= 0:
            raise ValueError("kd_scale must be positive")
        if action_bias_scale < 0:
            raise ValueError("action_bias_scale must be non-negative")
        if torque_bias_scale < 0:
            raise ValueError("torque_bias_scale must be non-negative")

        self._kp_scale_range = (1.0 / kp_scale, 1.0 * kp_scale)
        self._kd_scale_range = (1.0 / kd_scale, 1.0 * kd_scale)
        self._action_bias_scale = action_bias_scale
        self._torque_bias_scale = torque_bias_scale

    def get_actuator_name(self, joint_name: str) -> str:
        return f"{joint_name}_ctrl"

    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PyTree:
        # Sample per-joint scales and biases; keep fixed for the whole episode.
        num = physics_data.ctrl.shape[0]
        rng_kp, rng_kd, rng_action_bias, rng_torque_bias = jax.random.split(rng, 4)

        kp_low, kp_high = self._kp_scale_range
        kd_low, kd_high = self._kd_scale_range

        kp_scale = jax.random.uniform(rng_kp, (num,), minval=kp_low, maxval=kp_high)
        kd_scale = jax.random.uniform(rng_kd, (num,), minval=kd_low, maxval=kd_high)
        # Sample symmetric biases: uniform in [-scale, +scale]
        action_bias = jax.random.uniform(
            rng_action_bias, (num,), minval=-self._action_bias_scale, maxval=self._action_bias_scale
        )
        torque_bias = jax.random.uniform(
            rng_torque_bias, (num,), minval=-self._torque_bias_scale, maxval=self._torque_bias_scale
        )

        return {
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "action_bias": action_bias,
            "torque_bias": torque_bias,
        }

    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        # Compute control with per-episode gain scales, action bias, and torque bias.
        scaled_action = action * self.action_scale
        # Apply per-episode action bias.
        biased_action = scaled_action + actuator_state["action_bias"]

        pos_rng, tor_rng = jax.random.split(rng)
        current_pos = physics_data.qpos[7:]  # First 7 are always root pos.
        current_vel = physics_data.qvel[6:]  # First 6 are always root vel.

        # Add noise to the action (position target); curriculum level not used for stateful → pass 1.0
        target_position = self.action_noise.add_noise(biased_action, 1.0, pos_rng)
        target_velocity = jnp.zeros_like(action)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel

        kp_eff = self.kps * actuator_state["kp_scale"]
        kd_eff = self.kds * actuator_state["kd_scale"]

        ctrl = kp_eff * pos_delta + kd_eff * vel_delta
        ctrl = self.torque_noise.add_noise(ctrl, 1.0, tor_rng)
        ctrl = ctrl + actuator_state["torque_bias"]

        return jnp.clip(ctrl, -self.ctrl_clip, self.ctrl_clip), actuator_state


class PositionVelocityActuator(_PositionActuatorsBase, StatefulActuators):
    """MIT Cheetah-style actuator controller operating on both position and velocity with per-episode randomizations."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        pos_action_noise: Noise | None = None,
        vel_action_noise: Noise | None = None,
        torque_noise: Noise | None = None,
        action_scale: float = 1.0,
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        pos_action_bias_scale: float = 0.0,
        vel_action_bias_scale: float = 0.0,
        torque_bias_scale: float = 0.0,
    ) -> None:
        _PositionActuatorsBase.__init__(
            self,
            physics_model=physics_model,
            metadata=metadata,
            action_noise=pos_action_noise,
            torque_noise=torque_noise,
            action_scale=action_scale,
        )

        self.vel_action_noise = NoNoise() if vel_action_noise is None else vel_action_noise

        if kp_scale <= 0:
            raise ValueError("kp_scale must be positive")
        if kd_scale <= 0:
            raise ValueError("kd_scale must be positive")
        if pos_action_bias_scale < 0:
            raise ValueError("pos_action_bias_scale must be non-negative")
        if vel_action_bias_scale < 0:
            raise ValueError("vel_action_bias_scale must be non-negative")
        if torque_bias_scale < 0:
            raise ValueError("torque_bias_scale must be non-negative")

        self._kp_scale_range = (1.0 / kp_scale, 1.0 * kp_scale)
        self._kd_scale_range = (1.0 / kd_scale, 1.0 * kd_scale)
        self._pos_action_bias_scale = pos_action_bias_scale
        self._vel_action_bias_scale = vel_action_bias_scale
        self._torque_bias_scale = torque_bias_scale

    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PyTree:
        # Sample per-joint scales and biases; keep fixed for the whole episode.
        num = physics_data.ctrl.shape[0]
        rng_kp, rng_kd, rng_pos_bias, rng_vel_bias, rng_torque_bias = jax.random.split(rng, 5)

        kp_low, kp_high = self._kp_scale_range
        kd_low, kd_high = self._kd_scale_range

        kp_scale = jax.random.uniform(rng_kp, (num,), minval=kp_low, maxval=kp_high)
        kd_scale = jax.random.uniform(rng_kd, (num,), minval=kd_low, maxval=kd_high)
        # Sample symmetric biases: uniform in [-scale, +scale]
        pos_action_bias = jax.random.uniform(
            rng_pos_bias, (num,), minval=-self._pos_action_bias_scale, maxval=self._pos_action_bias_scale
        )
        vel_action_bias = jax.random.uniform(
            rng_vel_bias, (num,), minval=-self._vel_action_bias_scale, maxval=self._vel_action_bias_scale
        )
        torque_bias = jax.random.uniform(
            rng_torque_bias, (num,), minval=-self._torque_bias_scale, maxval=self._torque_bias_scale
        )

        return {
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "pos_action_bias": pos_action_bias,
            "vel_action_bias": vel_action_bias,
            "torque_bias": torque_bias,
        }

    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Get the control signal from the (position and velocity) action vector."""
        pos_rng, vel_rng, tor_rng = jax.random.split(rng, 3)

        current_pos = physics_data.qpos[7:]  # First 7 are always root pos.
        current_vel = physics_data.qvel[6:]  # First 6 are always root vel.

        # Extract position and velocity targets
        target_position = action[: len(current_pos)]
        target_velocity = action[len(current_pos) :]
        chex.assert_equal_shape([current_pos, target_position, current_vel, target_velocity])

        # Apply per-episode action biases
        target_position = target_position * self.action_scale + actuator_state["pos_action_bias"]
        target_velocity = target_velocity + actuator_state["vel_action_bias"]

        # Add position and velocity noise (curriculum level not used for stateful → pass 1.0)
        target_position = self.action_noise.add_noise(target_position, 1.0, pos_rng)
        target_velocity = self.vel_action_noise.add_noise(target_velocity, 1.0, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel

        kp_eff = self.kps * actuator_state["kp_scale"]
        kd_eff = self.kds * actuator_state["kd_scale"]

        ctrl = kp_eff * pos_delta + kd_eff * vel_delta
        ctrl = self.torque_noise.add_noise(ctrl, 1.0, tor_rng)
        ctrl = ctrl + actuator_state["torque_bias"]

        return jnp.clip(ctrl, -self.ctrl_clip, self.ctrl_clip), actuator_state

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        """Get the default action (zeros) with the correct shape."""
        qpos_dim = len(physics_data.qpos[7:])
        return jnp.zeros(qpos_dim * 2)

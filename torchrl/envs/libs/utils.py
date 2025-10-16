# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copied from gym > 0.19 release
#
# this file should only be accessed when gym is installed
from __future__ import annotations

import collections
import copy
import inspect

import numpy as np

IMPORT_ERROR = None
try:
    # rule of thumbs: gym precedes
    from gym import ObservationWrapper, spaces
except ImportError as err:
    IMPORT_ERROR = err
    try:
        from gymnasium import ObservationWrapper, spaces
    except ImportError as err2:
        raise err from err2

STATE_KEY = "observation"


class GymPixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values (Gym ≤0.25 and Gymnasium ≥0.26).

    - Old Gym: calls env.render(mode="rgb_array")
    - Gymnasium: expects env to be created with render_mode="rgb_array"
      and calls env.render() with no kwargs.
    """

    def __init__(self, env, pixels_only=True, render_kwargs=None, pixel_keys=("pixels",)):
        # reset first so render() has something to show
        env.reset()
        super().__init__(env)

        self._pixels_only = pixels_only
        self._pixel_keys = pixel_keys
        self._observation_is_dict = isinstance(env.observation_space, (spaces.Dict, collections.abc.MutableMapping))

        # Determine if this env.render supports a "mode" kwarg (old Gym) or not (Gymnasium).
        sig = None
        try:
            sig = inspect.signature(self.env.render)
            self._supports_mode = "mode" in sig.parameters
        except (TypeError, ValueError):
            # Some C extensions or wrappers don’t have inspectable signatures; fall back to try/except at call time.
            self._supports_mode = None

        if render_kwargs is None:
            render_kwargs = {}
        self._render_kwargs = {}
        for key in pixel_keys:
            kw = dict(render_kwargs.get(key, {}))
            if self._supports_mode is True:
                kw.setdefault("mode", "rgb_array")
            else:
                # Gymnasium path: ensure we don't pass "mode"
                kw.pop("mode", None)
            self._render_kwargs[key] = kw

        # Build new observation space
        if isinstance(env.observation_space, spaces.Box):
            invalid_keys = {STATE_KEY}
        elif self._observation_is_dict:
            invalid_keys = set(env.observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            overlapping = set(pixel_keys) & set(invalid_keys)
            if overlapping:
                raise ValueError(f"Duplicate or reserved pixel keys {overlapping!r}.")

        if pixels_only:
            obs_space = spaces.Dict()
        elif self._observation_is_dict:
            obs_space = copy.deepcopy(env.observation_space)
        else:
            obs_space = spaces.Dict()
            obs_space.spaces[STATE_KEY] = env.observation_space

        # Probe a frame to infer pixel shape/dtype and finalize the space
        pixels_spaces = {}
        for pixel_key in pixel_keys:
            frame = self._render_frame(pixel_key)
            if frame is None:
                raise RuntimeError(
                    "env.render() returned None. With Gymnasium (≥0.26), you must create the env with "
                    "render_mode='rgb_array' (e.g., gym.make(..., render_mode='rgb_array') or, for Minari, "
                    "ds.recover_environment(render_mode='rgb_array'))."
                )
            frame = np.asarray(frame)
            if np.issubdtype(frame.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(frame.dtype, np.floating):
                low, high = (-float("inf"), float("inf"))
            else:
                raise TypeError(f"Unsupported pixel dtype: {frame.dtype}")
            pixels_spaces[pixel_key] = spaces.Box(shape=frame.shape, low=low, high=high, dtype=frame.dtype)

        obs_space.spaces.update(pixels_spaces)
        self.observation_space = obs_space

    def _render_frame(self, pixel_key):
        kw = self._render_kwargs.get(pixel_key, {})
        # Try explicit kwargs first; if TypeError, fall back to no-kwargs (Gymnasium).
        try:
            return self.env.render(**kw) if kw else self.env.render()
        except TypeError:
            return self.env.render()

    def observation(self, wrapped_observation):
        # Build the base container
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(wrapped_observation)(wrapped_observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = wrapped_observation

        # Add pixels
        for pixel_key in self._pixel_keys:
            frame = self._render_frame(pixel_key)
            if frame is None:
                raise RuntimeError(
                    "env.render() returned None during step. Ensure the env was created with "
                    "render_mode='rgb_array'."
                )
            observation[pixel_key] = frame

        return observation

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import gymnasium as gym
import numpy as np
import torch.nn
import torch.optim
from gymnasium.spaces import Dict, Discrete, Box
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    EGreedyModule,
    MLP,
    ProbabilisticActor,
    QValueActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import CQLLoss, DiscreteCQLLoss, SoftUpdate
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.models import ACTIVATIONS

# ====================================================================
# Environment utils
# -----------------

# --- mission vocab (extend as needed) ---
VERBS  = ["open", "pick", "go"]          # add "put", "toggle", ...
NOUNS  = ["door", "key", "ball"]         # add others as needed
COLORS = ["red", "blue", "green", "yellow", "purple", "grey"]  # BabyAI colors

VERB2ID  = {w: i for i, w in enumerate(VERBS)}
NOUN2ID  = {w: i for i, w in enumerate(NOUNS)}
COLOR2ID = {w: i for i, w in enumerate(COLORS)}

UNK_VERB  = len(VERBS)
UNK_NOUN  = len(NOUNS)
UNK_COLOR = len(COLORS)


def parse_mission_parts(text: str):
    """Very dumb parser: lowercases and scans tokens for verb/noun/color. Returns (verb_id, noun_id, color_id)."""
    s = text.lower()
    tokens = s.replace(".", " ").replace(",", " ").split()
    v = next((VERB2ID[t] for t in tokens if t in VERB2ID), UNK_VERB)
    n = next((NOUN2ID[t] for t in tokens if t in NOUN2ID), UNK_NOUN)
    c = next((COLOR2ID[t] for t in tokens if t in COLOR2ID), UNK_COLOR)
    return v, n, c


class MissionSlotsTwoHot(gym.ObservationWrapper):
    """
    Replace 'mission' string with a 2-d one-hot [pick_key, open_door]
    AND return ONLY the keys we declare ('image', 'mission').
    This prevents TorchRL from seeing unexpected keys like 'direction'.
    """
    def __init__(self, env, default_open=True):
        super().__init__(env)
        assert isinstance(env.observation_space, Dict), \
            "Expected Dict observation with keys like {'image','mission'}."
        if "image" not in env.observation_space.spaces or "mission" not in env.observation_space.spaces:
            raise RuntimeError("MissionSlotsTwoHot expects 'image' and 'mission' in the base env observation_space.")

        img_space = env.observation_space.spaces["image"]
        self.default_open = default_open

        # We only expose these two keys
        self.keep_keys = ("image", "mission")

        self.observation_space = Dict(
            image=img_space,
            mission=Box(low=0, high=1, shape=(2,), dtype=np.uint8),
        )

    def observation(self, obs):
        # --- build the two-hot from the *original* mission ---
        s = obs["mission"]
        s = s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
        s = s.lower()
        vec = np.zeros(2, dtype=np.uint8)
        if ("pick" in s and "key" in s) or ("grab" in s and "key" in s):
            vec[0] = 1  # pick_key
        elif "open" in s and "door" in s:
            vec[1] = 1  # open_door
        else:
            vec[1 if self.default_open else 0] = 1

        # --- return ONLY the keys we promise in observation_space ---
        return {
            "image": obs["image"],
            "mission": vec,
        }


class FlattenConcatKeys(torch.nn.Module):
    """
    Flattens the first tensor (e.g., image) and any number of extra tensors,
    casts to float32, optionally normalizes uint8 to [0,1], and concatenates.

    Use via TensorDictModule with in_keys=[image_key, *extra_keys] and a single out_key.
    """
    def __init__(self, normalize_uint8_image: bool = False, normalize_uint8_extras: bool = False):
        super().__init__()
        self.normalize_uint8_image = normalize_uint8_image
        self.normalize_uint8_extras = normalize_uint8_extras

    @staticmethod
    def _prep(t: torch.Tensor, normalize: bool) -> torch.Tensor:
        # Cast + optional 0..255 -> 0..1
        if t.dtype == torch.uint8:
            t = t.to(torch.float32)
            if normalize:
                t = t / 255.0
        else:
            t = t.to(torch.float32)
        # Flatten all non-batch dims
        return t.flatten(1)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        assert len(tensors) >= 1, "FlattenConcatKeys expects at least one input tensor"
        img = self._prep(tensors[0], self.normalize_uint8_image)
        if len(tensors) == 1:
            return img
        extras = [self._prep(t, self.normalize_uint8_extras) for t in tensors[1:]]
        return torch.cat([img] + extras, dim=-1)


class ConcatImageWithMissionEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int = 2, emb_dim: int = 32, normalize_uint8_image: bool = False):
        super().__init__()
        self.normalize_uint8_image = normalize_uint8_image
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)

    def _prep_img(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if self.normalize_uint8_image and x.dtype == torch.uint8:
            x = x / 255.0
        return x.flatten(1)

    @staticmethod
    def _to_index(m: torch.Tensor) -> torch.Tensor:
        # supports one-hot [B,V] or indices [B]
        return m.argmax(-1) if (m.ndim >= 2 and m.shape[-1] > 1) else m.long()

    def forward(self, image: torch.Tensor, mission: torch.Tensor) -> torch.Tensor:
        img = self._prep_img(image)
        midx = self._to_index(mission)
        memb = self.emb(midx)  # [B, emb_dim]
        return torch.cat([img, memb], dim=-1)


class MissionPartsWrapper(gym.ObservationWrapper):
    """
    Returns observation as:
      {
        "image":   Box(..., dtype=uint8),
        "verb":    Discrete(|VERBS|+1 for UNK),
        "noun":    Discrete(|NOUNS|+1),
        "color":   Discrete(|COLORS|+1),
      }
    Filters out any extra keys (e.g., direction).
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Dict), "Expected Dict obs."
        img_space = env.observation_space.spaces["image"]
        self.observation_space = Dict({
            "image": img_space,
            "verb":  Discrete(len(VERBS)  + 1),
            "noun":  Discrete(len(NOUNS)  + 1),
            "color": Discrete(len(COLORS) + 1),
        })

    def observation(self, obs):
        s = obs["mission"]
        s = s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
        v, n, c = parse_mission_parts(s)
        return {
            "image": obs["image"],
            "verb":  np.int64(v),
            "noun":  np.int64(n),
            "color": np.int64(c),
        }


class ConcatImageWithFactorizedMissionEmbedding(torch.nn.Module):
    """
    Flattens image and concatenates with learned embeddings for verb/noun/color.
    Accepts each as either indices [B] or one-hot [B,V], returns a single tensor.
    """
    def __init__(self, verb_vocab: int, noun_vocab: int, color_vocab: int,
                 verb_dim: int = 16, noun_dim: int = 16, color_dim: int = 8,
                 normalize_uint8_image: bool = False, combiner: str = "concat"):
        super().__init__()
        self.normalize_uint8_image = normalize_uint8_image
        self.combiner = combiner  # "concat" or "sum"
        self.emb_v = torch.nn.Embedding(verb_vocab,  verb_dim)
        self.emb_n = torch.nn.Embedding(noun_vocab,  noun_dim)
        self.emb_c = torch.nn.Embedding(color_vocab, color_dim)

    def _prep_img(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if self.normalize_uint8_image and x.dtype == torch.uint8:  # add the flag
            x = x / 255.0
        return x.flatten(1)

    @staticmethod
    def _to_index(x: torch.Tensor) -> torch.Tensor:
        # supports indices [B] or one-hot [B,V]
        return x.argmax(-1) if (x.ndim >= 2 and x.shape[-1] > 1) else x.long()

    def forward(self, image: torch.Tensor, verb: torch.Tensor, noun: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        img = self._prep_img(image)
        vidx = self._to_index(verb)
        nidx = self._to_index(noun)
        cidx = self._to_index(color)

        ev, en, ec = self.emb_v(vidx), self.emb_n(nidx), self.emb_c(cidx)

        if self.combiner == "sum":
            # pad to same dim then sum
            maxd = max(ev.shape[-1], en.shape[-1], ec.shape[-1])
            if ev.shape[-1] != maxd:
                ev = torch.nn.functional.pad(ev, (0, maxd - ev.shape[-1]))
            if en.shape[-1] != maxd:
                en = torch.nn.functional.pad(en, (0, maxd - en.shape[-1]))
            if ec.shape[-1] != maxd:
                ec = torch.nn.functional.pad(ec, (0, maxd - ec.shape[-1]))
            memb = ev + en + ec
        else:
            memb = torch.cat([ev, en, ec], dim=-1)

        return torch.cat([img, memb], dim=-1)  # <- a single tensor


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(cfg.env.name, device=device, from_pixels=from_pixels, pixels_only=False)

    elif lib == "dm_control":
        env = DMControlEnv(cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False)
        return TransformedEnv(env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation"))

    elif lib == "minari":
        import minari, warnings, gymnasium as gym
        ds = minari.load_dataset(cfg.replay_buffer.dataset)

        want_pixels = bool(getattr(cfg.logger, "video", False)) or from_pixels

        # 1) Try to recover with render_mode (Gymnasium style)
        rec_kwargs = {"render_mode": "rgb_array"} if want_pixels else {}
        try:
            base_env = ds.recover_environment(eval_env=False, **rec_kwargs)
        except TypeError:
            # older Minari signature: fall back and patch below
            base_env = ds.recover_environment(eval_env=False)

        # 2) If we still don't have rgb_array, rebuild via gym.make using ds.env_spec
        if want_pixels and getattr(base_env, "render_mode", None) != "rgb_array":
            spec = getattr(ds, "env_spec", None)
            try:
                env_id = getattr(spec, "id", None) or (str(spec) if spec is not None else None)
                kwargs = dict(getattr(spec, "kwargs", {}) or {})
                kwargs["render_mode"] = "rgb_array"
                if env_id is None:
                    raise RuntimeError("Minari dataset env_spec missing/unknown.")
                try:
                    base_env.close()
                except Exception:
                    pass
                base_env = gym.make(env_id, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"Failed to remake env with render_mode='rgb_array': {e}. "
                    "Video will be disabled for this run."
                )
                want_pixels = False  # avoid pixel wrapper -> avoid crash

        # 3) Your mission parsing wrapper
        env = MissionPartsWrapper(base_env)

        # 4) Hand instance to TorchRL. Let GymWrapper add pixels only if we *know* the env can render.
        return GymWrapper(env, device=device, from_pixels=want_pixels, pixels_only=False)

    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(
    env,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, train_num_envs=1, eval_num_envs=1, logger=None):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, cfg)
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.optim.seed)

    train_env = apply_env_transforms(parallel_env)

    maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    eval_env.set_seed(cfg.optim.seed)
    if cfg.logger.video:
        eval_env = eval_env.insert_transform(
            0, VideoRecorder(logger=logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(
    cfg,
    train_env,
    actor_model_explore,
    compile=False,
    compile_mode=None,
    cudagraph=False,
):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile else False,
        cudagraph_policy=cudagraph,
    )
    collector.set_seed(getattr(cfg.optim, "seed", 0))
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


def make_offline_replay_buffer(rb_cfg):
    data = MinariExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=True),
        prefetch=4,
        download=True,
    )

    data.append_transform(DoubleToFloat())

    return data


def make_offline_discrete_replay_buffer(rb_cfg):
    import gymnasium as gym
    import minari
    from minari import DataCollector

    # Create custom minari dataset from environment

    env = gym.make(rb_cfg.env)
    env = DataCollector(env)

    for _ in range(rb_cfg.episodes):
        env.reset(seed=123)
        while True:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.create_dataset(
        dataset_id=rb_cfg.dataset,
        algorithm_name="Random-Policy",
        code_permalink="https://github.com/Farama-Foundation/Minari",
        author="Farama",
        author_email="contact@farama.org",
    )

    data = MinariExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        load_from_local_minari=True,
        sampler=SamplerWithoutReplacement(drop_last=True),
        prefetch=4,
    )

    data.append_transform(DoubleToFloat())

    # Clean up
    minari.delete_dataset(rb_cfg.dataset)

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


def make_cql_model(cfg, train_env, eval_env, device="cpu"):
    model_cfg = cfg.model

    action_spec = train_env.action_spec_unbatched

    actor_net, q_net = make_cql_modules_state(model_cfg, eval_env)
    in_keys = ["observation"]
    out_keys = ["loc", "scale"]

    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=out_keys)

    # We use a ProbabilisticActor to make sure that we map the
    # network output to the right space using a TanhDelta
    # distribution.
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        # Wrapping the kwargs in a TensorDictParams such that these items are
        #  send to device when necessary - not compatible with compile yet
        # distribution_kwargs=TensorDictParams(
        #     TensorDict(
        #         {
        #             "low": torch.as_tensor(action_spec.space.low, device=device),
        #             "high": torch.as_tensor(action_spec.space.high, device=device),
        #             "tanh_loc": NonTensorData(False),
        #         }
        #     ),
        #     no_convert=True,
        # ),
        distribution_kwargs={
            "low": action_spec.space.low.to(device),
            "high": action_spec.space.high.to(device),
            "tanh_loc": False,
        },
        default_interaction_type=ExplorationType.RANDOM,
    )

    in_keys = ["observation", "action"]

    out_keys = ["state_action_value"]
    qvalue = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=q_net,
    )

    model = torch.nn.ModuleList([actor, qvalue]).to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model


def make_discretecql_model(cfg, train_env, eval_env, device="cpu"):
    model_cfg = cfg.model
    action_spec = train_env.action_spec

    # ---- NEW: configurable concat of image + arbitrary extra keys ----
    concat_cfg = getattr(model_cfg, "concat", None)
    image_key = (concat_cfg.image_key if concat_cfg and "image_key" in concat_cfg else "image")
    extra_keys = list(concat_cfg.extra_keys) if (concat_cfg and "extra_keys" in concat_cfg) else ["mission"]
    out_key = (concat_cfg.out_key if concat_cfg and "out_key" in concat_cfg else "obs_vec")
    norm_img = bool(getattr(concat_cfg, "normalize_uint8_image", False)) if concat_cfg else False
    norm_ext = bool(getattr(concat_cfg, "normalize_uint8_extras", False)) if concat_cfg else False

    in_keys_for_concat = [image_key] + (extra_keys or [])

    # pre = TensorDictModule(
    #     FlattenConcatKeys(
    #         normalize_uint8_image=norm_img,
    #         normalize_uint8_extras=norm_ext,
    #     ),
    #     in_keys=in_keys_for_concat,
    #     out_keys=[out_key],
    # )

    # pre = TensorDictModule(
    #     ConcatImageWithMissionEmbedding(
    #         vocab_size=getattr(model_cfg, "mission_vocab_size", 2),
    #         emb_dim=getattr(model_cfg, "mission_emb_dim", 32),
    #         normalize_uint8_image=bool(getattr(model_cfg.get("concat", {}), "normalize_uint8_image", False)),
    #     ),
    #     in_keys=[image_key, "mission"],  # <- tensors extracted from TD
    #     out_keys=[out_key],  # <- single tensor returned goes here
    # )

    pre = TensorDictModule(
        ConcatImageWithFactorizedMissionEmbedding(
            verb_vocab=len(VERBS) + 1,
            noun_vocab=len(NOUNS) + 1,
            color_vocab=len(COLORS) + 1,
            verb_dim=getattr(model_cfg, "verb_emb_dim", 16),
            noun_dim=getattr(model_cfg, "noun_emb_dim", 16),
            color_dim=getattr(model_cfg, "color_emb_dim", 8),
            normalize_uint8_image=bool(getattr(model_cfg.get("concat", {}), "normalize_uint8_image", False)),
            combiner=getattr(model_cfg, "mission_combiner", "concat"),
        ),
        in_keys=["image", "verb", "noun", "color"],
        out_keys=[out_key],  # e.g., "obs_vec"
    )

    # ---- Q network on the concatenated vector ----
    actor_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    actor_net = MLP(**actor_net_kwargs)

    # Inner QValueActor that reads the concatenated vector
    qvalue_core = QValueActor(
        module=actor_net,
        spec=Composite(action=action_spec),
        in_keys=[out_key],
    )

    # Wrap preprocessor + qvalue_core
    qvalue_net = TensorDictSequential(pre, qvalue_core).to(device)

    # ---- expose attributes needed by DiscreteCQLLoss on the wrapper ----
    qvalue_net.action_space = "categorical"
    # optional but nice to have
    if hasattr(qvalue_core, "spec"):
        qvalue_net.spec = qvalue_core.spec

    # init
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset().to(device)
        qvalue_net(td)

    del td
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=action_spec,
    )
    model_explore = TensorDictSequential(qvalue_net, greedy_module).to(device)
    return qvalue_net, model_explore


def make_cql_modules_state(model_cfg, proof_environment):
    action_spec = proof_environment.action_spec_unbatched

    actor_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    actor_net = MLP(**actor_net_kwargs)
    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{model_cfg.default_policy_scale}",
        scale_lb=model_cfg.scale_lb,
    )
    actor_net = torch.nn.Sequential(actor_net, actor_extractor)

    qvalue_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": 1,
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }

    q_net = MLP(**qvalue_net_kwargs)

    return actor_net, q_net


# ====================================================================
# CQL Loss
# ---------


def make_continuous_loss(loss_cfg, model, device: torch.device | None = None):
    loss_module = CQLLoss(
        model[0],
        model[1],
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        min_q_weight=loss_cfg.min_q_weight,
        max_q_backup=loss_cfg.max_q_backup,
        deterministic_backup=loss_cfg.deterministic_backup,
        num_random=loss_cfg.num_random,
        with_lagrange=loss_cfg.with_lagrange,
        lagrange_thresh=loss_cfg.lagrange_thresh,
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma, device=device)
    target_net_updater = SoftUpdate(loss_module, tau=loss_cfg.tau)

    return loss_module, target_net_updater


def make_discrete_loss(loss_cfg, model, device: torch.device | None = None):

    if "action_space" in loss_cfg:  # especify action space
        loss_module = DiscreteCQLLoss(
            model,
            loss_function=loss_cfg.loss_function,
            action_space=loss_cfg.action_space,
            delay_value=True,
        )
    else:
        loss_module = DiscreteCQLLoss(
            model,
            loss_function=loss_cfg.loss_function,
            delay_value=True,
        )

    loss_module.make_value_estimator(gamma=loss_cfg.gamma, device=device)
    target_net_updater = SoftUpdate(loss_module, tau=loss_cfg.tau)

    return loss_module, target_net_updater


def make_discrete_cql_optimizer(cfg, loss_module):
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    return optim


def make_continuous_cql_optimizer(cfg, loss_module):
    critic_params = loss_module.qvalue_network_params.flatten_keys().values()
    actor_params = loss_module.actor_network_params.flatten_keys().values()
    actor_optim = torch.optim.Adam(
        actor_params,
        lr=cfg.optim.actor_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    critic_optim = torch.optim.Adam(
        critic_params,
        lr=cfg.optim.critic_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    alpha_optim = torch.optim.Adam(
        [loss_module.log_alpha],
        lr=cfg.optim.actor_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    if loss_module.with_lagrange:
        alpha_prime_optim = torch.optim.Adam(
            [loss_module.log_alpha_prime],
            lr=cfg.optim.critic_lr,
        )
    else:
        alpha_prime_optim = None
    return actor_optim, critic_optim, alpha_optim, alpha_prime_optim


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

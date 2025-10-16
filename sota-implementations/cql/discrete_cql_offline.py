# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CQL Example.

This is a self-contained example of a discrete offline CQL training script.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from tensordict.nn import CudaGraphModule

from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_discrete_cql_optimizer,
    make_discrete_loss,
    make_discretecql_model,
    make_environment,
    make_offline_replay_buffer,
    parse_mission_parts,
)

torch.set_float32_matmul_precision("high")


# ---- robust mission two-hot for replay batches ----
def _encode_mission_twohot_inplace(td, default_open=True):
    """
    Encode BOTH ('observation','mission') and ('next','observation','mission')
    to 2-d two-hot [pick_key, open_door]. Robust to NonTensorStack, lists, scalars.
    """
    import torch

    def _as_list(x):
        # Convert NonTensorStack / object tensor / ndarray / list / scalar into a Python list
        if isinstance(x, torch.Tensor) and x.dtype == torch.object:
            arr = x.cpu().numpy()
            if arr.shape == ():
                return [arr.item()]
            return arr.tolist()
        if isinstance(x, np.ndarray):
            return [x.item()] if x.shape == () else x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        # NonTensorStack or other iterable (not bytes/str)
        if hasattr(x, "__len__") and not isinstance(x, (bytes, bytearray, str)):
            try:
                return list(x)
            except Exception:
                return [x]
        return [x]

    def _encode_list(missions, B):
        # fix length to B (replicate or truncate)
        if len(missions) == 1 and B > 1:
            missions = missions * B
        elif len(missions) != B:
            missions = [missions[i % len(missions)] for i in range(B)]
        # two-hot encode
        out = []
        for m in missions:
            s = m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
            s = s.lower()
            v = torch.zeros(2, dtype=torch.uint8)
            if ("pick" in s and "key" in s) or ("grab" in s and "key" in s):
                v[0] = 1
            elif "open" in s and "door" in s:
                v[1] = 1
            else:
                v[1 if default_open else 0] = 1
            out.append(v)
        return torch.stack(out, 0)  # [B,2], uint8

    # infer batch size
    bs = td.batch_size
    B = int(bs[0]) if len(bs) else 1

    # 1) observation.mission
    try:
        val = td.get(("observation", "mission"))
        # if already numeric two-hot with proper batch, skip
        if isinstance(val, torch.Tensor) and val.dtype != torch.object and val.ndim >= 2 and val.shape[0] == B and val.shape[-1] == 2:
            pass
        else:
            missions = _as_list(val)
            batch = _encode_list(missions, B)
            td.set(("observation", "mission"), batch)
    except KeyError:
        pass

    # 2) next.observation.mission (keep specs consistent)
    try:
        val_n = td.get(("next", "observation", "mission"))
        if isinstance(val_n, torch.Tensor) and val_n.dtype != torch.object and val_n.ndim >= 2 and val_n.shape[0] == B and val_n.shape[-1] == 2:
            pass
        else:
            missions_n = _as_list(val_n)
            batch_n = _encode_list(missions_n, B)
            td.set(("next", "observation", "mission"), batch_n)
    except KeyError:
        pass

    return td


def _encode_mission_parts_inplace(td):
    import torch

    def _to_list(x):
        if isinstance(x, torch.Tensor) and x.dtype == torch.object:
            arr = x.cpu().numpy()
            return arr.tolist() if arr.shape else [arr.item()]
        if hasattr(x, "__iter__") and not isinstance(x, (bytes, bytearray, str)):
            try: return list(x)
            except Exception: return [x]
        return [x]

    B = int(td.batch_size[0]) if len(td.batch_size) else 1
    for path in [("observation","mission"), ("next","observation","mission")]:
        try:
            val = td.get(path)
        except KeyError:
            continue
        missions = _to_list(val)
        if len(missions) == 1 and B > 1:
            missions = missions * B
        v_idx, n_idx, c_idx = [], [], []
        for m in missions:
            s = m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
            v, n, c = parse_mission_parts(s)
            v_idx.append(v); n_idx.append(n); c_idx.append(c)
        td.set(path[:-1] + ("verb",),  torch.tensor(v_idx, dtype=torch.int64))
        td.set(path[:-1] + ("noun",),  torch.tensor(n_idx, dtype=torch.int64))
        td.set(path[:-1] + ("color",), torch.tensor(c_idx, dtype=torch.int64))
    # mirror to top-level for current step
    for k in ("verb","noun","color","image"):
        try:
            td.set(k, td.get(("observation", k)))
        except KeyError:
            pass
    # mirror into next sub-TD
    for k in ("verb","noun","color","image"):
        try:
            td.set(("next", k), td.get(("next","observation", k)))
        except KeyError:
            pass
    return td


@hydra.main(version_base="1.1", config_path="", config_name="minari_discrete_config")
def main(cfg):  # noqa: F821
    device = cfg.optim.device
    if device in ("", None):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("DiscreteCQL", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="discretecql_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    seed = getattr(cfg.optim, "seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create env
    train_env, eval_env = make_environment(
        cfg, train_num_envs=1, eval_num_envs=cfg.logger.eval_envs, logger=logger
    )

    # Create agent
    model, explore_policy = make_discretecql_model(cfg, train_env, eval_env, device)

    del train_env

    # Create loss
    loss_module, target_net_updater = make_discrete_loss(cfg.loss, model, device)

    # Create optimizers
    optimizer = make_discrete_cql_optimizer(cfg, loss_module)  # optimizer for CQL loss

    def update(data):

        # Compute loss components
        loss_vals = loss_module(data)

        q_loss = loss_vals["loss_qvalue"]
        cql_loss = loss_vals["loss_cql"]

        # Total loss = Q-learning loss + CQL regularization
        loss = q_loss + cql_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Soft update of target Q-network
        target_net_updater.step()

        # Detach to avoid keeping computation graph in logging
        return loss.detach(), loss_vals.detach()

    compile_mode = None
    if cfg.compile.compile:
        if cfg.compile.compile_mode not in (None, ""):
            compile_mode = cfg.compile.compile_mode
        elif cfg.compile.cudagraphs:
            compile_mode = "default"
        else:
            compile_mode = "reduce-overhead"
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            """
            CudaGraphModule is experimental and may silently lead to incorrect results.
            Use with caution.
            """,
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    gradient_steps = cfg.optim.gradient_steps
    policy_eval_start = cfg.optim.policy_eval_start
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    # ---- optional mission encoding toggle (from cfg.mission.*) ----
    # Defaults: don't encode; default class = "open_door"
    mission_encode = False
    default_open = True
    try:
        if "mission" in cfg and cfg.mission is not None:
            mission_encode = bool(getattr(cfg.mission, "encode_replay", False))
            # Either mission.default or mission.twohot.default may be present
            default_val = None
            if hasattr(cfg.mission, "default"):
                default_val = str(cfg.mission.default).lower()
            elif hasattr(cfg.mission, "twohot") and "default" in cfg.mission.twohot:
                default_val = str(cfg.mission.twohot.default).lower()
            if default_val in ("pick_key", "pick", "key"):
                default_open = False
    except Exception:
        # keep safe defaults
        pass

    # Training loop
    policy_eval_start = torch.tensor(policy_eval_start, device=device)
    for i in range(gradient_steps):
        timeit.printevery(1000, gradient_steps, erase=True)
        pbar.update(1)
        # sample data
        with timeit("sample"):
            data = replay_buffer.sample()
            # data = _encode_mission_twohot_inplace(data, default_open=True)
            data = _encode_mission_parts_inplace(data)  # instead of two-hot

        with timeit("update"):
            torch.compiler.cudagraph_mark_step_begin()
            loss, loss_vals = update(data.to(device))

        # log metrics
        metrics_to_log = {
            "loss": loss.cpu(),
            **loss_vals.cpu(),
        }

        # evaluation
        with timeit("log/eval"):
            if i % evaluation_interval == 0:
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ), torch.no_grad():
                    eval_td = eval_env.rollout(
                        max_steps=eval_steps,
                        policy=explore_policy,
                        auto_cast_to_device=True,
                    )
                    eval_env.apply(dump_video)

                # eval_td: matrix of shape: [num_episodes, max_steps, ...]
                eval_reward = (
                    eval_td["next", "reward"].sum(1).mean().item()
                )  # mean computed over the sum of rewards for each episode
                metrics_to_log["evaluation_reward"] = eval_reward

        with timeit("log"):
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, i)

    pbar.close()
    if not eval_env.is_closed:
        eval_env.close()


if __name__ == "__main__":
    main()

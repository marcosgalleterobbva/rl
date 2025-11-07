# debug.py
from __future__ import annotations
import os, json, math, logging
from pathlib import Path
from dataclasses import dataclass, field
import torch
import numpy as np

log = logging.getLogger("cqldebug")
log.setLevel(logging.INFO)

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _safe_td_get(td, key_tuple, default=None):
    try:
        return td.get(key_tuple)
    except KeyError:
        return default

def _to_list_any(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.object:
        arr = x.cpu().numpy()
        return arr.tolist() if arr.shape else [arr.item()]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

@dataclass
class DebugCfg:
    enable: bool = False
    freq: int = 1000
    save_dir: str = "debug_local"
    log_batch_samples: int = 2
    checks: list[str] = field(default_factory=list)

class DebugSuite:
    def __init__(self, cfg) -> None:
        # cfg is Hydra’s node; pull fields with defaults
        self.cfg = DebugCfg(
            enable=bool(getattr(cfg.debug, "enable", False)) if "debug" in cfg else False,
            freq=int(getattr(cfg.debug, "freq", 1000)) if "debug" in cfg else 1000,
            save_dir=str(getattr(cfg.debug, "save_dir", "debug_local")) if "debug" in cfg else "debug_local",
            log_batch_samples=int(getattr(cfg.debug, "log_batch_samples", 2)) if "debug" in cfg else 2,
            checks=list(getattr(cfg.debug, "checks", [])) if "debug" in cfg else [],
        )
        _ensure_dir(self.cfg.save_dir)
        self.step0_snapshot_done = False
        self.have_video = False

    # ------------ lifecycle hooks -----------------
    def on_start(self, replay_buffer, train_env, eval_env, model, loss_module):
        if not self.cfg.enable: return
        log.info("[debug] on_start")
        # Dataset head
        if "dataset_head" in self.cfg.checks:
            try:
                head = replay_buffer[:8]  # tiny slice
                self._dump_json("dataset_head_shapes.json", self._td_shapes(head))
                # sample print
                missions = _safe_td_get(head, ("observation","mission"))
                if missions is not None:
                    ml = _to_list_any(missions)[:self.cfg.log_batch_samples]
                    log.info(f"[dataset_head] missions sample: {ml}")
                actions = head.get("action") if "action" in head.keys(True, True) else None
                if actions is not None:
                    vals, cnt = torch.unique(actions.flatten(), return_counts=True)
                    log.info(f"[dataset_head] action hist: {dict(zip(vals.tolist(), cnt.tolist()))}")
            except Exception as e:
                log.warning(f"[dataset_head] failed: {e}")

        # Quick smoke on eval env reset
        td = eval_env.reset()
        for k in ("image","verb","noun","color"):
            if k in td.keys():
                log.info(f"[env_reset] found key '{k}' with shape {tuple(td[k].shape)}")
        self.have_video = "pixels" in td.keys()

        # Model & loss smoke
        if "model_forward" in self.cfg.checks or "loss_forward" in self.cfg.checks:
            try:
                # tiny fake batch from eval reset
                bt = td.clone()
                bt.batch_size = torch.Size([1])
                out = model(bt.clone())
                assert torch.isfinite(out["action_value"]).all(), "NaNs in action_value"
                log.info(f"[model_forward] OK. action_value {tuple(out['action_value'].shape)}")
                if "loss_forward" in self.cfg.checks:
                    # Need a minimal loss call: add 'action' if missing
                    if "action" not in bt.keys():
                        # pick the greedy action
                        a = out["action_value"].argmax(-1)
                        bt.set("action", a)
                    l = loss_module(bt)
                    for k, v in l.items():
                        if torch.is_tensor(v):
                            assert torch.isfinite(v).all(), f"NaN in loss term {k}"
                    log.info("[loss_forward] OK.")
            except Exception as e:
                log.warning(f"[model/loss smoke] failed: {e}")

    def on_batch(self, step: int, data, model, loss_vals=None):
        if not (self.cfg.enable and self.cfg.freq and step % self.cfg.freq == 0):
            return

        # 1) Encoding invariants on missions/parts
        if "encoding_invariants" in self.cfg.checks:
            self._check_encoding(data)

        # 2) Preprocessor smoke: check obs_vec shape if present, image ranges
        if "preprocessor_smoke" in self.cfg.checks:
            self._check_preprocessor_inputs(data)

        # 3) Behavior cloning probe + q_gap
        if "bc_probe" in self.cfg.checks:
            self._bc_probe(step, data, model)

        # 4) Next keys exist
        if "next_keys" in self.cfg.checks:
            self._check_next_keys(data)

        # 5) Gradient health (if loss_vals were just computed)
        if "grad_health" in self.cfg.checks:
            self._check_grads(model)

        # 6) Dump one snapshot once
        if not self.step0_snapshot_done:
            self._dump_td("batch_step0.pt", data)
            self.step0_snapshot_done = True

    def on_eval(self, step: int, eval_td):
        if not self.cfg.enable: return
        if "eval_rollout_smoke" in self.cfg.checks:
            try:
                R = eval_td.get(("next","reward")).sum(1).mean().item()
                T = eval_td.get(("next","done")).shape[1] if ("next","done") in eval_td.keys(True, True) else None
                img = eval_td.get(("next","image"), None)
                if img is not None:
                    var = img.float().var().item()
                    log.info(f"[eval_rollout] reward_mean={R:.3f}, steps={T}, pixels_var={var:.3f}")
                else:
                    log.info(f"[eval_rollout] reward_mean={R:.3f}, steps={T}")
                # simple progression check: first vs last frame diff if image exists
                if img is not None and img.ndim >= 4:
                    first = img[:,0].float().mean().item()
                    last  = img[:,-1].float().mean().item()
                    log.info(f"[eval_rollout] first_px_mean={first:.2f}, last_px_mean={last:.2f}")
            except Exception as e:
                log.warning(f"[eval_rollout_smoke] failed: {e}")

    # ------------ individual checks -----------------
    def _check_encoding(self, td):
        B = int(td.batch_size[0]) if len(td.batch_size) else 1
        for path in [("observation","verb"), ("observation","noun"), ("observation","color"),
                     ("verb",), ("noun",), ("color",)]:
            x = _safe_td_get(td, path)
            if x is None: continue
            assert x.dtype in (torch.int64, torch.int32), f"{path} must be int indices"
            assert x.shape[0] == B, f"{path} batch mismatch: {x.shape} vs {B}"
            assert torch.all(x >= 0), f"{path} has negative indices"
        # sample print
        try:
            verbs = _safe_td_get(td, ("verb",))
            nouns = _safe_td_get(td, ("noun",))
            colors = _safe_td_get(td, ("color",))
            if verbs is not None and nouns is not None:
                log.info(f"[encoding] sample v/n/c: {verbs[:self.cfg.log_batch_samples].tolist()} "
                         f"{nouns[:self.cfg.log_batch_samples].tolist()} "
                         f"{colors[:self.cfg.log_batch_samples].tolist() if colors is not None else 'None'}")
        except Exception:
            pass

    def _check_preprocessor_inputs(self, td):
        img = _safe_td_get(td, ("image",))
        if img is None: return
        if img.dtype == torch.uint8:
            mi, ma = int(img.min()), int(img.max())
            log.info(f"[preproc] uint8 image min/max = {mi}/{ma} (hint: normalize_uint8_image==true?)")
        else:
            mi, ma = float(img.min()), float(img.max())
            log.info(f"[preproc] float image min/max = {mi:.3f}/{ma:.3f}")

    def _bc_probe(self, step, data, model):
        with torch.no_grad():
            td = data.clone()
            out = model(td)
            q = out["action_value"]
            a = td["action"].long().flatten()
            bc_acc = (q.argmax(-1) == a).float().mean().item()
            q_gap = (q.gather(-1, a.unsqueeze(-1)).squeeze(-1) - q.mean(-1)).mean().item()
            log.info(f"[bc_probe] step={step} bc_acc={bc_acc:.3f} q_gap={q_gap:.3f} "
                     f"q_mean={q.mean().item():.3f} q_std={q.std().item():.3f}")

    def _check_next_keys(self, td):
        for k in ("image","verb","noun","color"):
            v = _safe_td_get(td, ("next", k))
            if v is None:
                log.warning(f"[next_keys] missing ('next','{k}') in batch")

    def _check_grads(self, model):
        tot = 0.0
        big = 0
        for p in model.parameters():
            if p.grad is None: continue
            g = p.grad
            if not torch.isfinite(g).all():
                raise RuntimeError("Found NaN/Inf grads")
            n = g.norm().item()
            tot += n
            if n > 10.0:
                big += 1
        log.info(f"[grad_health] total_grad_norm~{tot:.2f}, big_tensors(>10)={big}")

    # ------------ helpers / dumps -----------------
    def _td_shapes(self, td):
        out = {}
        for k in td.keys(True, True):
            try:
                v = td.get(k)
                if torch.is_tensor(v):
                    out[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                else:
                    out[str(k)] = str(type(v))
            except Exception as e:
                out[str(k)] = f"error: {e}"
        return out

    def _dump_json(self, name, obj):
        p = Path(self.cfg.save_dir) / name
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)

    def _dump_td(self, name, td):
        p = Path(self.cfg.save_dir) / name
        try:
            torch.save(td.cpu(), p)
            log.info(f"[debug] saved {p}")
        except Exception as e:
            log.warning(f"[debug] failed to save TD: {e}")

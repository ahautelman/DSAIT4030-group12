from __future__ import annotations

from typing import Callable, Tuple

import torch


def _resolve_path_component(target, component: str):
    if component == "":
        return target

    try:
        index = int(component)
    except ValueError:
        if not hasattr(target, component):
            raise AttributeError(f"Could not resolve attribute '{component}' on {type(target).__name__}.")
        return getattr(target, component)

    try:
        return target[index]
    except Exception as exc:  # noqa: BLE001 - explicit, user-facing failure path
        raise IndexError(f"Could not index into {type(target).__name__} with component '{component}'.") from exc


def register_feature_hook(model: torch.nn.Module, hook_target_name: str) -> Tuple[Callable[[], torch.Tensor], torch.utils.hooks.RemovableHandle]:
    if not hook_target_name:
        raise ValueError("hook_target_name must be a non-empty string.")

    target = model
    for component in hook_target_name.split("."):
        target = _resolve_path_component(target, component)

    if not hasattr(target, "register_forward_hook"):
        raise TypeError(
            f"Resolved hook target '{hook_target_name}' on {type(model).__name__} does not support forward hooks."
        )

    state = {"ready": False, "value": None}

    def _hook(_module, _inputs, output):
        state["value"] = output[0] if isinstance(output, tuple) else output
        state["ready"] = True

    handle = target.register_forward_hook(_hook)

    def extractor_fn() -> torch.Tensor:
        if not state["ready"] or state["value"] is None:
            raise RuntimeError(
                f"No feature tensor has been captured yet for hook target '{hook_target_name}'. Run a forward pass first."
            )
        value = state["value"]
        assert isinstance(value, torch.Tensor)
        return value

    return extractor_fn, handle



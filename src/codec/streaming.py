from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch import nn


@dataclass
class StreamingState:
    batch_size: int
    device: torch.device

    def reset(self) -> None:
        pass


StateT = TypeVar("StateT", bound=StreamingState)


class StreamingModule(nn.Module, Generic[StateT]):
    """Small state-management API adapted from Mimi's streaming modules.

    This version intentionally omits Mimi's exec_mask and CUDAGraph support. 
    """

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: StateT | None = None
        self._cached_named_streaming_children: list[tuple[str, StreamingModule]] | None = None

    @property
    def is_streaming(self) -> bool:
        return self._streaming_state is not None

    def _apply_named_streaming(self, fn: Any) -> None:
        if self._cached_named_streaming_children is None:
            modules: list[tuple[str, StreamingModule]] = []

            def _handle_module(prefix: str, module: nn.Module) -> None:
                if isinstance(module, StreamingModule):
                    modules.append((prefix, module))
                for name, child in module.named_children():
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    _handle_module(child_prefix, child)

            _handle_module("", self)
            self._cached_named_streaming_children = modules
        for name, module in self._cached_named_streaming_children:
            fn(name, module)

    def streaming(self, batch_size: int) -> ExitStack:
        stack = ExitStack()

        def start(name: str, module: StreamingModule) -> None:
            if module._streaming_state is not None:
                raise RuntimeError(f"{name or module.__class__.__name__} is already streaming")
            module._streaming_state = module._init_streaming_state(batch_size)

        self._apply_named_streaming(start)
        stack.callback(self._stop_streaming)
        return stack

    def streaming_forever(self, batch_size: int) -> None:
        self.streaming(batch_size).__enter__()

    def _stop_streaming(self) -> None:
        def stop(name: str, module: StreamingModule) -> None:
            _ = name
            module._streaming_state = None

        self._apply_named_streaming(stop)

    def reset_streaming(self) -> None:
        def reset(name: str, module: StreamingModule) -> None:
            if module._streaming_state is None:
                raise RuntimeError(f"{name or module.__class__.__name__} is not streaming")
            module._streaming_state.reset()

        self._apply_named_streaming(reset)

    def get_streaming_state(self) -> dict[str, Any]:
        state = {}

        def add(name: str, module: StreamingModule) -> None:
            state[name] = module._streaming_state

        self._apply_named_streaming(add)
        return state

    def set_streaming_state(self, state: dict[str, Any]) -> None:
        state = dict(state)

        def set_state(name: str, module: StreamingModule) -> None:
            if name not in state:
                raise RuntimeError(f"Expected to find a streaming state for {name}.")
            module._streaming_state = state.pop(name)

        self._apply_named_streaming(set_state)
        if state:
            raise RuntimeError(f"Some states were not consumed: {list(state.keys())}")

    def _init_streaming_state(self, batch_size: int) -> StateT:
        raise NotImplementedError(f"{self.__class__.__name__} must implement _init_streaming_state().")


class StreamingContainer(StreamingModule[StreamingState]):
    def _init_streaming_state(self, batch_size: int) -> StreamingState:
        param = next(self.parameters(), None)
        if param is None:
            device = torch.device("cpu")
        else:
            device = param.device
        return StreamingState(batch_size, device)

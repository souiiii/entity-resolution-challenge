from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def ordered_map(function: Callable[[Any], Any], jobs: Iterable[tuple[Any, Any]], workers: int,
                window: int | None = None) -> Iterator[tuple[Any, Any]]:
    """Yield ``(context, function(argument))`` in input order with a bounded number of pending jobs."""
    if workers <= 1:
        for context, argument in jobs:
            yield context, function(argument)
        return
    window = window or 2 * workers
    with ProcessPoolExecutor(workers) as pool:
        pending: deque = deque()
        for context, argument in jobs:
            pending.append((context, pool.submit(function, argument)))
            if len(pending) >= window:
                context, future = pending.popleft()
                yield context, future.result()
        while pending:
            context, future = pending.popleft()
            yield context, future.result()

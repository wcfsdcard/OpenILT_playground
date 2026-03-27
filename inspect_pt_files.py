#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterator, Tuple

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch is required. Install it first, then run this script again.")
    sys.exit(1)


def iter_tensors(obj: Any, prefix: str = "root") -> Iterator[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_tensors(value, f"{prefix}[{repr(key)}]")
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from iter_tensors(value, f"{prefix}[{idx}]")
        return

    if isinstance(obj, set):
        for idx, value in enumerate(sorted(obj, key=repr)):
            yield from iter_tensors(value, f"{prefix}[{idx}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tensor shape and dtype for .pt files."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="./kernel/kernels",
        help="Directory containing .pt files (default: ./kernel/kernels)",
    )
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help='Glob pattern for files to inspect (default: "*.pt")',
    )
    args = parser.parse_args()

    root = Path(args.directory)
    files = sorted(root.glob(args.pattern))

    if not files:
        print(f"No files matching {args.pattern!r} found in {root}")
        return

    for path in files:
        print(path)
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception as exc:
            print(f"  [load error] {exc}")
            continue

        tensors = list(iter_tensors(obj))
        if not tensors:
            print(f"  no tensors found (top-level type: {type(obj).__name__})")
            continue

        for name, tensor in tensors:
            print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()

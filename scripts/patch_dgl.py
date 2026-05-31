"""Post-install patch for DGL graphbolt compatibility.

DGL 2.2.0's graphbolt module fails on import when:
1. The C++ .dylib doesn't match the installed PyTorch version
2. torchdata.datapipes is missing (deprecated in newer torchdata)

This script patches graphbolt/__init__.py to gracefully catch those errors.
Run after `uv sync` or whenever DGL is reinstalled.
"""

import os
import sys

GRAPHBOLT_INIT = """\"\"\"Graphbolt.\"\"\"
import os
import sys

import torch

from .._ffi import libinfo


def load_graphbolt():
    \"\"\"Load Graphbolt C++ library\"\"\"
    vers = torch.__version__.split("+", maxsplit=1)[0]

    if sys.platform.startswith("linux"):
        basename = f"libgraphbolt_pytorch_{vers}.so"
    elif sys.platform.startswith("darwin"):
        basename = f"libgraphbolt_pytorch_{vers}.dylib"
    elif sys.platform.startswith("win"):
        basename = f"graphbolt_pytorch_{vers}.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)

    dirname = os.path.dirname(libinfo.find_lib_path()[0])
    path = os.path.join(dirname, "graphbolt", basename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find DGL C++ graphbolt library at {path}"
        )

    try:
        torch.classes.load_library(path)
    except Exception:  # pylint: disable=W0703
        raise ImportError("Cannot load Graphbolt C++ library")


try:
    load_graphbolt()
    # pylint: disable=wrong-import-position
    from .base import *
    from .minibatch import *
    from .dataloader import *
    from .dataset import *
    from .feature_fetcher import *
    from .feature_store import *
    from .impl import *
    from .itemset import *
    from .item_sampler import *
    from .minibatch_transformer import *
    from .negative_sampler import *
    from .sampled_subgraph import *
    from .subgraph_sampler import *
    from .internal import (
        compact_csc_format,
        unique_and_compact,
        unique_and_compact_csc_formats,
    )
    from .utils import add_reverse_edges, exclude_seed_edges
except (FileNotFoundError, ImportError, ModuleNotFoundError):
    pass
"""


def main():
    try:
        import dgl
    except ImportError:
        print("DGL not installed, nothing to patch.")
        return

    init_path = os.path.join(os.path.dirname(dgl.__file__), "graphbolt", "__init__.py")

    if not os.path.exists(init_path):
        print(f"graphbolt __init__.py not found at {init_path}")
        return

    with open(init_path) as f:
        content = f.read()

    if "except (FileNotFoundError, ImportError, ModuleNotFoundError):" in content:
        print("DGL graphbolt already patched.")
        return

    with open(init_path, "w") as f:
        f.write(GRAPHBOLT_INIT)

    print(f"Patched {init_path}")


if __name__ == "__main__":
    main()

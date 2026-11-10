import importlib
from pathlib import Path
import sys


FRAMEWORK_ROOT = Path(__file__).resolve().parents[2] / "third_party/instattention"


def load_framework():
    package_root = (FRAMEWORK_ROOT / "flexgen").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "flexgen" or name.startswith("flexgen."):
            filename = getattr(module, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(package_root):
                raise RuntimeError(f"refusing external FlexGen module {name}: {filename}")
    if str(FRAMEWORK_ROOT) in sys.path:
        sys.path.remove(str(FRAMEWORK_ROOT))
    sys.path.insert(0, str(FRAMEWORK_ROOT))
    module = importlib.import_module("flexgen.flex_opt")
    if Path(module.__file__).resolve() != package_root / "flex_opt.py":
        raise RuntimeError("FlexGen did not resolve to the CEMU source copy")
    return module

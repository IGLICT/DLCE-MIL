"""Model package exports.

Note:
- `model_interfacedepth.py` may be experimental / optional.
- The stable Lightning entrypoint used by `train.py` is `model_interface.py`.
"""

# Prefer the stable interface.
from .model_interface import ModelInterface

# If you later restore a depth-specific interface, you can switch back via:
# from .model_interfacedepth import ModelInterface
# from .depth_wrapper import DepthAwareWrapper
# from .transmil_depth import TransMIL  # 如果你打算以后用这个新文件里的 TransMIL
# from .depth_wrapper_dtype_safe import DepthAwareWrapper 
# from .depth_wrapper_dtype_safe import RefinedKernel3Wrapper

# Optional explicit export (keeps namespace clean for controlled experiments)
# Keep optional so `import models` won't crash when `transmil_depth.py` is broken
# or not needed by the current YAML.
try:
	from .transmil_depth import TransmilDepth
except Exception:
	TransmilDepth = None

# Optional: bias version selectable via config `Model.name: transmil_bias`
from .transmil_bias import TransmilBias

# Optional: hybrid model combining add2 relation attention and prototype pooling
try:
	from .TransMILadd2Prototype import TransMILadd2Prototype, TransMILAdd2Prototype, TransMILHybrid
except Exception:
	TransMILadd2Prototype = None
	TransMILAdd2Prototype = None
	TransMILHybrid = None
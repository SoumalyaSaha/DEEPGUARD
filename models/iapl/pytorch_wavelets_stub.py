import sys
import types


class DWTForward(types.SimpleNamespace):
    pass


class DWTInverse(types.SimpleNamespace):
    pass


def install_stub():
    """Inject a stand-in 'pytorch_wavelets' module into sys.modules.

    models/iapl/iapl_vendor/{clip_models,freq_stem}.py import DWTForward/DWTInverse
    at module load time, but never instantiate them in the plain single-image
    forward path (verified: freq_stem.ConvNet.forward does not touch DWT; the
    UFD-style freq/DCT branches don't use wavelets in CLIPModel.forward).
    The real package needs a C++ build step and is not in the repo's
    requirements.txt, so we stub the import names instead.
    """
    if "pytorch_wavelets" in sys.modules and hasattr(sys.modules["pytorch_wavelets"], "DWTForward"):
        return
    m = types.ModuleType("pytorch_wavelets")
    m.DWTForward = DWTForward
    m.DWTInverse = DWTInverse
    sys.modules["pytorch_wavelets"] = m
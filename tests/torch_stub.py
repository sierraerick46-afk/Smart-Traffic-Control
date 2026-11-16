"""Minimal stand-in for PyTorch so the saved scikit-learn model can be loaded in environments
without torch (CI, quick local checks). Never used by the real application."""
import sys
import types


def install():
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")

    class Tensor:  # noqa: D401
        pass

    class Module:
        def __init__(self, *a, **k):
            pass

    nn = types.ModuleType("torch.nn")
    nn.Module = Module
    for n in ("Sequential", "Linear", "ReLU", "Dropout", "MSELoss", "Identity"):
        setattr(nn, n, type(n, (Module,), {}))
    functional = types.ModuleType("torch.nn.functional")
    optim = types.ModuleType("torch.optim")
    torch.Tensor = Tensor
    torch.nn = nn
    torch.optim = optim
    nn.functional = functional
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": functional, "torch.optim": optim})

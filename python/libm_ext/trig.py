from libm_ext import _core


def sinpi(x: float) -> float:
    """
    Compute $\sin(\pi x)$ more accurately than `sin(pi*x)`, especially for large `x`.
    """
    return _core.sinpi(x)


def cospi(x: float) -> float:
    """
    Compute $\cos(\pi x)$ more accurately than `cos(pi*x)`, especially for large `x`.
    """
    return _core.cospi(x)


def sincospi(x: float) -> tuple[float, float]:
    """
    Simultaneously compute `sinpi` and `cospi`.
    """
    return _core.sincospi(x)

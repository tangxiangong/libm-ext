from libm_ext import _core
import numpy as np
import numpy.typing as npt


def _sinpi_serial(
    x: npt.NDArray,
) -> npt.NDArray:
    """
    vectorized sinpi using serial mode.
    """
    if x.dtype == np.float64:
        return _core.sinpi_vectorized_serial(x)
    elif x.dtype == np.float32:
        return _core.sinpif_vectorized_serial(x)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def _sinpi_parallel(
    x: npt.NDArray,
) -> npt.NDArray:
    """
    vectorized sinpi using parallel mode.
    """
    if x.dtype == np.float64:
        return _core.sinpi_vectorized_parallel(x)
    elif x.dtype == np.float32:
        return _core.sinpif_vectorized_parallel(x)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def sinpi(x: npt.NDArray, threshold: int = 10000) -> npt.NDArray:
    """
    vectorized sinpi using threshold mode.

    Args:
        x: input array
        threshold: threshold for switching to parallel mode
    Returns:
        sinpi(x)

    Raises:
        ValueError: if dtype of x is not supported (float64 or float32)
    """
    if x.size < threshold:
        return _sinpi_serial(x)
    else:
        return _sinpi_parallel(x)


def _cospi_serial(
    x: npt.NDArray,
) -> npt.NDArray:
    """
    vectorized cospi using serial mode.
    """
    if x.dtype == np.float64:
        return _core.cospi_vectorized_serial(x)
    elif x.dtype == np.float32:
        return _core.cospif_vectorized_serial(x)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def _cospi_parallel(
    x: npt.NDArray,
) -> npt.NDArray:
    """
    vectorized cospi using parallel mode.
    """
    if x.dtype == np.float64:
        return _core.cospi_vectorized_parallel(x)
    elif x.dtype == np.float32:
        return _core.cospif_vectorized_parallel(x)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def cospi(x: npt.NDArray, threshold: int = 16000) -> npt.NDArray:
    """
    vectorized cospi using threshold mode.

    Args:
        x: input array
        threshold: threshold for switching to parallel mode
    Returns:
        cospi(x)

    Raises:
        ValueError: if dtype of x is not supported (float64 or float32)
    """
    if x.size < threshold:
        return _cospi_serial(x)
    else:
        return _cospi_parallel(x)


if __name__ == "__main__":
    from libm_ext.ext.utils import bench

    x = np.random.standard_normal(16000)
    bench("sinpi-serial", lambda: _sinpi_serial(x))
    bench("sinpi-parallel", lambda: _sinpi_parallel(x))

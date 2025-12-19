//! Trigonometric functions
//!
//! The implementation is based on the [Julia Standard Library](https://github.com/JuliaLang/julia/blob/master/base/special/trig.jl).
//!
//! - [`sinpi`]
//! - [`cospi`]
//! - [`sinpif`]
//! - [`cospif`]
//! - [`sincospi`]
//! - [`sincospif`]
//!

use crate::utils::evalpoly;

const MAXINTFLOAT64: u64 = 1 << f64::MANTISSA_DIGITS;
const MAXINTFLOAT32: u64 = 1 << f32::MANTISSA_DIGITS;

/// Uses minimax polynomial of $\sin(\pi x)$ for $\pi x \in [0, 0.25]$ .
#[inline]
fn sinpi_kernel(x: f64) -> f64 {
    let x_square = x * x;
    let x_bisquare = x_square * x_square;
    let coes = [
        2.5501640398773415,
        -0.5992645293202981,
        0.08214588658006512,
        -7.370429884921779e-3,
        4.662827319453555e-4,
        -2.1717412523382308e-5,
    ];
    let r = evalpoly(&coes, x_square);
    let inner = x_bisquare.mul_add(r, 1.2245907532225998e-16);
    let inner = (-5.16771278004997f64).mul_add(x_square, inner);

    std::f64::consts::PI.mul_add(x, x * inner)
}

#[inline]
fn sinpif_kernel(x: f32) -> f32 {
    sinpif_kernel_wide(x) as f32
}

/// Uses minimax polynomial of $\cos(\pi x)$ for $\pi x \in [0, 0.25]$ .
#[inline]
fn cospi_kernel(x: f64) -> f64 {
    let x_square = x * x;
    let coes = [
        4.058712126416765,
        -1.3352627688537357,
        0.23533063027900392,
        -0.025806887811869204,
        1.9294917136379183e-3,
        -1.0368935675474665e-4,
    ];
    let r = x_square * evalpoly(&coes, x_square);
    let a_x_square = 4.934802200544679 * x_square;
    let a_x_square_lo = 3.109686485461973e-16f64.mul_add(
        x_square,
        4.934802200544679f64.mul_add(x_square, -a_x_square),
    );
    let w = 1.0 - a_x_square;

    w + x_square.mul_add(r, ((1.0 - w) - a_x_square) - a_x_square_lo)
}

#[inline]
fn cospif_kernel(x: f32) -> f32 {
    cospif_kernel_wide(x) as f32
}

#[inline]
fn cospif_kernel_wide(x: f32) -> f64 {
    let x_f64 = x as f64;
    let coes = [
        1.0,
        -4.934802200541122,
        4.058712123568637,
        -1.3352624040152927,
        0.23531426791507182,
        -0.02550710082498761,
    ];
    evalpoly(&coes, x_f64 * x_f64)
}

#[inline]
fn sinpif_kernel_wide(x: f32) -> f64 {
    let x_f64 = x as f64;
    let coes = [
        std::f64::consts::PI,
        -5.167712769188119,
        2.5501626483206374,
        -0.5992021090314925,
        0.08100185277841528,
    ];
    x_f64 * evalpoly(&coes, x_f64 * x_f64)
}

/// Compute $\sin(\pi x)$ more accurately than `sin(pi*x)`, especially for large `x` (f64).
///
/// # Notes
///
/// If `x` is infinite or NAN, return NAN.
pub fn sinpi(x: f64) -> f64 {
    let x_abs = x.abs();
    if x_abs.is_infinite() || x_abs.is_nan() {
        return f64::NAN;
    }
    // If x is too large, return 0.0
    if x_abs >= MAXINTFLOAT64 as f64 {
        return 0.0f64.copysign(x);
    }

    // reduce x to interval [0, 0.5]
    let n = (2.0 * x_abs).round();
    let rx = (-0.5f64).mul_add(n, x_abs);
    let n = n as i64 & 3;
    let res = match n {
        0 => sinpi_kernel(rx),
        1 => cospi_kernel(rx),
        2 => 0.0 - sinpi_kernel(rx),
        _ => 0.0 - cospi_kernel(rx),
    };
    if x.is_sign_negative() { -res } else { res }
}

/// Compute $\sin(\pi x)$ more accurately than `sin(pi*x)`, especially for large `x` (f32).
///
/// # Notes
///
/// If `x` is infinite or NAN, return NAN.
pub fn sinpif(x: f32) -> f32 {
    let x_abs = x.abs();
    if x_abs.is_infinite() || x_abs.is_nan() {
        return f32::NAN;
    }
    // If x is too large, return 0.0
    if x_abs >= MAXINTFLOAT32 as f32 {
        return 0.0f32.copysign(x);
    }

    // reduce x to interval [0, 0.5]
    let n = (2.0 * x_abs).round();
    let rx = (-0.5f32).mul_add(n, x_abs);
    let n = n as i64 & 3;
    let res = match n {
        0 => sinpif_kernel(rx),
        1 => cospif_kernel(rx),
        2 => 0.0 - sinpif_kernel(rx),
        _ => 0.0 - cospif_kernel(rx),
    };
    if x.is_sign_negative() { -res } else { res }
}

/// Compute $\cos(\pi x)$ more accurately than `cos(pi*x)`, especially for large `x` (f64).
///
/// # Notes
///
/// If `x` is infinite or NAN, return NAN.
pub fn cospi(x: f64) -> f64 {
    let x_abs = x.abs();
    if x_abs.is_infinite() || x_abs.is_nan() {
        return f64::NAN;
    }
    // If x is too large, return 1.0
    if x_abs >= MAXINTFLOAT64 as f64 {
        return 1.0;
    }

    // reduce x to interval [0, 0.5]
    let n = (2.0 * x_abs).round();
    let rx = (-0.5f64).mul_add(n, x_abs);
    let n = n as i64 & 3;
    match n {
        0 => cospi_kernel(rx),
        1 => 0.0 - sinpi_kernel(rx),
        2 => 0.0 - cospi_kernel(rx),
        _ => sinpi_kernel(rx),
    }
}

/// Compute $\cos(\pi x)$ more accurately than `cos(pi*x)`, especially for large `x` (f32).
///
/// # Notes
///
/// If `x` is infinite or NAN, return NAN.
pub fn cospif(x: f32) -> f32 {
    let x_abs = x.abs();
    if x_abs.is_infinite() || x_abs.is_nan() {
        return f32::NAN;
    }
    // If x is too large, return 1.0
    if x_abs >= MAXINTFLOAT32 as f32 {
        return 1.0;
    }

    // reduce x to interval [0, 0.5]
    let n = (2.0 * x_abs).round();
    let rx = (-0.5f32).mul_add(n, x_abs);
    let n = n as i64 & 3;
    match n {
        0 => cospif_kernel(rx),
        1 => 0.0 - sinpif_kernel(rx),
        2 => 0.0 - cospif_kernel(rx),
        _ => sinpif_kernel(rx),
    }
}

/// Simultaneously compute [`sinpi`] and [`cospi`] (f64).
///
/// # Notes
///
/// If `x` is infinite or NAN, return (NAN, NAN).
pub fn sincospi(x: f64) -> (f64, f64) {
    let x_abs = x.abs();
    if x_abs.is_infinite() || x_abs.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    // If x is too large, return 0.0
    if x_abs >= MAXINTFLOAT64 as f64 {
        return (0.0f64.copysign(x), 1.0);
    }

    // reduce x to interval [0, 0.5]
    let n = (2.0 * x_abs).round();
    let rx = (-0.5f64).mul_add(n, x_abs);
    let n = n as i64 & 3;
    let mut si = sinpi_kernel(rx);
    let mut co = cospi_kernel(rx);
    (si, co) = match n {
        0 => (si, co),
        1 => (co, 0.0 - si),
        2 => (0.0 - si, 0.0 - co),
        _ => (0.0 - co, si),
    };
    si = if x.is_sign_negative() { -si } else { si };
    (si, co)
}

/// Simultaneously compute [`sinpif`] and [`cospif`] (f32).
///
/// # Notes
///
/// If `x` is infinite or NAN, return (NAN, NAN).
pub fn sincospif(x: f32) -> (f32, f32) {
    let x_abs = x.abs();
    if x_abs.is_infinite() || x_abs.is_nan() {
        return (f32::NAN, f32::NAN);
    }
    // If x is too large, return 0.0
    if x_abs >= MAXINTFLOAT32 as f32 {
        return (0.0f32.copysign(x), 1.0f32);
    }

    // reduce x to interval [0, 0.5]
    let n = (2.0 * x_abs).round();
    let rx = (-0.5f32).mul_add(n, x_abs);
    let n = n as i64 & 3;
    let mut si = sinpif_kernel(rx);
    let mut co = cospif_kernel(rx);
    (si, co) = match n {
        0 => (si, co),
        1 => (co, 0.0 - si),
        2 => (0.0 - si, 0.0 - co),
        _ => (0.0 - co, si),
    };
    si = if x.is_sign_negative() { -si } else { si };
    (si, co)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::assert_approx_eq;

    const EPSILON_F64: f64 = 1e-15;
    const EPSILON_F32: f32 = 1e-6;

    #[test]
    fn test_maxintfloat64() {
        assert_eq!(9007199254740992, MAXINTFLOAT64);
    }

    #[test]
    fn test_maxintfloat32() {
        assert_eq!(16777216, MAXINTFLOAT32);
    }

    #[test]
    fn test_sinpi_special_values() {
        // sin(pi * n) = 0 for integer n
        for i in -10..=10 {
            let x = i as f64;
            assert_eq!(sinpi(x), 0.0, "sinpi({}) should be 0.0", x);
            // Check sign of zero preservation if needed, though 0.0 vs -0.0 can be tricky
        }

        // sin(pi * (n + 0.5)) = (-1)^n
        assert_approx_eq!(sinpi(0.5), 1.0, EPSILON_F64);
        assert_approx_eq!(sinpi(1.5), -1.0, EPSILON_F64);
        assert_approx_eq!(sinpi(-0.5), -1.0, EPSILON_F64);
    }

    #[test]
    fn test_cospi_special_values() {
        // cos(pi * n) = (-1)^n
        assert_approx_eq!(cospi(0.0), 1.0, EPSILON_F64);
        assert_approx_eq!(cospi(1.0), -1.0, EPSILON_F64);
        assert_approx_eq!(cospi(2.0), 1.0, EPSILON_F64);
        assert_approx_eq!(cospi(-1.0), -1.0, EPSILON_F64);

        // cos(pi * (n + 0.5)) = 0
        assert_approx_eq!(cospi(0.5), 0.0, EPSILON_F64);
        assert_approx_eq!(cospi(1.5), 0.0, EPSILON_F64);
        assert_approx_eq!(cospi(-0.5), 0.0, EPSILON_F64);
    }

    #[test]
    fn test_sincospi_consistency() {
        let values = [-0.1, 0.2, 0.33, 0.5, 10.7, 1000.123];
        for &x in &values {
            let (s, c) = sincospi(x);
            let s_single = sinpi(x);
            let c_single = cospi(x);
            assert_approx_eq!(s, s_single, EPSILON_F64);
            assert_approx_eq!(c, c_single, EPSILON_F64);
        }
    }

    #[test]
    fn test_pythagorean_identity() {
        // sin^2 + cos^2 = 1
        let values = [0.123, 0.456, 1.789, -2.345, 100.0];
        for &x in &values {
            let s = sinpi(x);
            let c = cospi(x);
            assert_approx_eq!(s * s + c * c, 1.0, EPSILON_F64);
        }
    }

    #[test]
    fn test_nan_inf() {
        assert!(sinpi(f64::NAN).is_nan());
        assert!(sinpi(f64::INFINITY).is_nan());
        assert!(sinpi(f64::NEG_INFINITY).is_nan());

        assert!(cospi(f64::NAN).is_nan());
        assert!(cospi(f64::INFINITY).is_nan());
    }

    // f32 tests
    #[test]
    fn test_sinpif_special_values() {
        assert_approx_eq!(sinpif(0.0), 0.0, EPSILON_F32);
        assert_approx_eq!(sinpif(0.5), 1.0, EPSILON_F32);
        assert_approx_eq!(sinpif(1.0), 0.0, EPSILON_F32);
    }

    #[test]
    fn test_cospif_special_values() {
        assert_approx_eq!(cospif(0.0), 1.0, EPSILON_F32);
        assert_approx_eq!(cospif(0.5), 0.0, EPSILON_F32);
        assert_approx_eq!(cospif(1.0), -1.0, EPSILON_F32);
    }
}

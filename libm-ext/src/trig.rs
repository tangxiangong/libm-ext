//! Trigonometric functions
//!
//! The implementation is based on the [Julia Standard Library](https://github.com/JuliaLang/julia/blob/master/base/special/trig.jl).
//!
//! - [`sinpi`] / [`sinpif`]
//! - [`cospi`] / [`cospif`]
//! - [`sincospi`] / [`sincospif`]
//!
#![allow(clippy::approx_constant)]

use crate::utils::evalpoly;
use hexf::hexf64;

const MAXINTFLOAT64: u64 = 1 << f64::MANTISSA_DIGITS;
const MAXINTFLOAT32: u64 = 1 << f32::MANTISSA_DIGITS;

/// Uses minimax polynomial of $\sin(\pi x)/x$ for $x \in [0, 0.25]$ to approximate $\sin(\pi x)$.
#[inline]
fn sinpi_kernel(x: f64) -> f64 {
    // coefficients for minimax polynormal of sin(pi*x)/x
    // c0 = c0_hi + c0_lo
    let c0_hi = hexf64!("0x1.921fb54442d18p+1"); // 3.141592653589793
    let c0_lo = hexf64!("0x1.1a5f14401a52p-53"); // 1.2245907532226012e-16
    let c2 = hexf64!("-0x1.4abbce625be01p2"); // -5.167712780049897
    let c4 = hexf64!("0x1.466bc67753d55p1"); // 2.550164039864891
    let c6 = hexf64!("-0x1.32d2ccde1222cp-1"); // -0.5992645283777782
    let c8 = hexf64!("0x1.50782aae65ef4p-4"); // 8.214584991798884e-2
    let c10 = hexf64!("-0x1.e2fa787f268fep-8"); // -7.369665544638913e-3
    let c12 = hexf64!("0x1.e06afd55415bp-12"); // 4.5816223912739217e-4
    let c14 = hexf64!("0x1.add9d45ae8195p-17"); // 1.2810554997078747e-5

    let x_square = x * x;
    let x_bisquare = x_square * x_square;

    // c4 + c6*x^2 + ... + c14*x^10
    let mut r = evalpoly(&[c4, c6, c8, c10, c12, c14], x_square);
    // c0_lo + c4*x^4 + c6*x^6 + ... + c14*x^14
    r = x_bisquare.mul_add(r, c0_lo);
    // c0_lo + c2*x^2 + c4*x^4 + ... + c14*x^14
    r = c2.mul_add(x_square, r);
    // c0_hi*x + co_lo*x + c2*x^3 + c4*x^5 + ... + c14*x^15
    c0_hi.mul_add(x, x * r)
}

#[inline]
fn sinpif_kernel(x: f32) -> f32 {
    let c0 = hexf64!("0x1.921fb6p1"); // 3.1415927410125732
    let c2 = hexf64!("-0x1.4abc1cp2"); // -5.167731285095215
    let c4 = hexf64!("0x1.468e6cp1"); // 2.5512213706970215
    let c6 = hexf64!("-0x1.3e497cp-1"); // -0.6216543912887573
    let c8 = hexf64!("0x1.eb5482p-3"); // 0.23990727961063385

    let x_f64 = x as f64;

    let res_f64 = x_f64 * evalpoly(&[c0, c2, c4, c6, c8], x_f64 * x_f64);
    res_f64 as f32
}

/// Uses minimax polynomial of $\cos(\pi x)$ for $x \in [0, 0.25]$.
///
/// # Double-Double Compensation Technique
///
/// To achieve nearly 1 ULP accuracy, we use double-double arithmetic for the
/// leading terms where cancellation error is most significant.
///
/// The polynomial approximation is:
/// $$\cos(\pi x) \approx c_0 + c_2 x^2 + c_4 x^4 + \cdots + c_{14} x^{14}$$
///
/// where $c_0 = 1$ and $c_2 \approx -\pi^2/2 \approx -4.9348$.
///
/// ## Why Double-Double?
///
/// When computing $1 + c_2 x^2$ for small $x$, the result is close to 1, causing
/// significant cancellation. A single f64 cannot represent $c_2$ exactly, so we
/// split it: $c_2 = c_{2,hi} - c_{2,lo}$ where $c_{2,hi}$ is the nearest f64 and
/// $c_{2,lo}$ is the small correction term.
///
/// ## Algorithm
///
/// 1. Compute `a_x_square_hi = c2_hi * x²` (primary term)
/// 2. Compute `a_x_square_lo = c2_lo * x² + rounding_error(c2_hi * x²)`
///    using FMA to capture the rounding error: `(-c2_hi) * x² + a_x_square_hi`
/// 3. Compute `w = c0 + a_x_square_hi` (may lose precision)
/// 4. Recover lost precision: `(c0 - w) + a_x_square_hi` gives the rounding error
/// 5. Final result compensates: `w + (higher_terms + rounding_error - c2_lo*x²)`
///
/// This technique is based on Dekker's TwoSum and TwoProduct algorithms.
#[inline]
fn cospi_kernel(x: f64) -> f64 {
    // coefficients for minimax polynomial of cos(pi*x)
    let c0 = 1.0;
    // c2 = c2_hi - c2_lo
    let c2_hi = hexf64!("-0x1.3bd3cc9be45dep2"); // -4.934802200544679
    let c2_lo = hexf64!("0x1.219c35926754dp-52"); // 2.5119679985578543e-16
    let c4 = hexf64!("0x1.03c1f081b5a67p2"); // 4.058712126416686
    let c6 = hexf64!("-0x1.55d3c7e3c325bp0"); // -1.3352627688465393
    let c8 = hexf64!("0x1.e1f5067b90b37p-3"); // 0.23533062996474438
    let c10 = hexf64!("-0x1.a6d1e7294bffap-6"); // -2.5806880706284098e-2
    let c12 = hexf64!("0x1.f9c89ca1d5187p-10"); // 1.9294114685709256e-3
    let c14 = hexf64!("-0x1.b167302e37198p-14"); // -1.0333134625590266e-4

    let x_square = x * x;

    // Higher-order terms: c4*x^4 + c6*x^6 + ... + c14*x^14
    let r = x_square * evalpoly(&[c4, c6, c8, c10, c12, c14], x_square);

    // Double-double multiplication: c2 * x² = (c2_hi - c2_lo) * x²
    // Step 1: Primary product
    let a_x_square_hi = c2_hi * x_square;
    // Step 2: Capture rounding error of c2_hi * x² using FMA, plus c2_lo * x²
    //         = c2_lo * x² + (c2_hi * x² - a_x_square_hi)  [rounding error]
    let a_x_square_lo = c2_lo.mul_add(x_square, (-c2_hi).mul_add(x_square, a_x_square_hi));

    // Double-double addition: c0 + c2_hi * x²
    let w = c0 + a_x_square_hi;

    // Compensated summation:
    // - (c0 - w) + a_x_square_hi: recovers rounding error from c0 + a_x_square_hi
    // - Subtract a_x_square_lo: because c2 = c2_hi - c2_lo, we need to subtract c2_lo * x²
    w + x_square.mul_add(r, ((c0 - w) + a_x_square_hi) - a_x_square_lo)
}

#[inline]
fn cospif_kernel(x: f32) -> f32 {
    let c0 = 1.0;
    let c2 = hexf64!("-0x1.3bd3ccp2"); // -4.934802055358887
    let c4 = hexf64!("0x1.03c1a6p2"); // 4.058694362640381
    let c6 = hexf64!("-0x1.55a3b4p0"); // -1.334529161453247
    let c8 = hexf64!("0x1.c85d38p-3"); // 0.222834050655365
    let c10 = hexf64!("0x1.97cb1p-5"); // 0.04977944493293762

    let x_f64 = x as f64;

    let res_f64 = evalpoly(&[c0, c2, c4, c6, c8, c10], x_f64 * x_f64);
    res_f64 as f32
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

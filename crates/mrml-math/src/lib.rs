#![no_std]

#[cfg_attr(unix, link(name = "m"))]
#[cfg_attr(windows, link(name = "msvcrt"))]
unsafe extern "C" {
    #[link_name = "sqrtf"]
    fn native_sqrt(value: f32) -> f32;
    #[link_name = "powf"]
    fn native_pow(value: f32, exponent: f32) -> f32;
    #[link_name = "cosf"]
    fn native_cos(value: f32) -> f32;
    #[link_name = "sinf"]
    fn native_sin(value: f32) -> f32;
    #[link_name = "tanhf"]
    fn native_tanh(value: f32) -> f32;
    #[link_name = "expf"]
    fn native_exp(value: f32) -> f32;
    #[link_name = "roundf"]
    fn native_round(value: f32) -> f32;
}

#[inline]
pub fn sqrt(value: f32) -> f32 {
    unsafe { native_sqrt(value) }
}
#[inline]
pub fn pow(value: f32, exponent: f32) -> f32 {
    unsafe { native_pow(value, exponent) }
}
#[inline]
pub fn cos(value: f32) -> f32 {
    unsafe { native_cos(value) }
}
#[inline]
pub fn sin(value: f32) -> f32 {
    unsafe { native_sin(value) }
}
#[inline]
pub fn tanh(value: f32) -> f32 {
    unsafe { native_tanh(value) }
}
#[inline]
pub fn exp(value: f32) -> f32 {
    unsafe { native_exp(value) }
}
#[inline]
pub fn round(value: f32) -> f32 {
    unsafe { native_round(value) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_math_matches_known_values() {
        assert!((super::sqrt(9.0) - 3.0).abs() < 1e-6);
        assert!((super::pow(2.0, 3.0) - 8.0).abs() < 1e-6);
        assert!((super::sin(0.0)).abs() < 1e-6);
        assert!((super::cos(0.0) - 1.0).abs() < 1e-6);
        assert_eq!(super::round(2.6), 3.0);
    }
}

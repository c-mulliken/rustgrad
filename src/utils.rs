pub fn approx(a: f64, b: f64, thresh: f64) -> bool {
    (a - b).abs() < thresh
}

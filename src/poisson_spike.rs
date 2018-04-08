use rand::{Rng, XorShiftRng};


pub fn generate_spike(fr: f64, dt: f64, t: f64, rng: &mut XorShiftRng) -> Vec<u8> {
    let num = (t/dt) as usize;
    let v: Vec<u8> = rng.gen_iter().take(num).map(|x: f64| if x < fr*dt { 1 } else { 0 } ).collect();
    v
}
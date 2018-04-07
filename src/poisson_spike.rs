use rand::{thread_rng, Rng};


pub fn generate_spike(fr: f64, dt: f64, t: f64) -> Vec<u8> {
    let num = (t/dt) as usize;
    let v: Vec<u8> = thread_rng().gen_iter().take(num).map(|x: f64| if x < fr*dt { 1 } else { 0 } ).collect();
    v
}
use rand::{Rng, SeedableRng, XorShiftRng};

fn rk4<F: Fn(f64)->f64>(f: F, y: f64, dt: f64) -> f64 {
    let k1 = dt * f(y);
    let k2 = dt * f(y + k1*0.5);
    let k3 = dt * f(y + k2*0.5);
    let k4 = dt * f(y + k3);
    (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
}

pub struct Neuron {
    pub v: f64,
    pub e_l: f64,
    pub theta: f64,
    pub dt: f64,
    pub weights: Vec<f64>,
    pub refractory_state: usize,
    pub refractory_period: usize
}

impl Neuron {
    pub fn new(v_init: f64, e_l: f64, theta: f64, dt: f64, num_inputs: usize, refactory_period: usize, rng: &mut XorShiftRng) -> Neuron {
        let weights: Vec<f64> = rng.gen_iter().take(num_inputs).collect();
        Neuron {
            v: v_init,
            e_l: e_l,
            theta: theta,
            dt: dt,
            weights: weights,
            refractory_period: refactory_period,
            refractory_state: 0
        }
    }

    pub fn run(&mut self, input: &Vec<u8>) -> (u8, f64) {
        let mut sum_v = 0.0;

        for i in 0..input.len() {
            sum_v += self.weights[i]*(input[i] as f64);
        }

        let e_l = self.e_l;
        let d_v = |y:f64| (e_l - y + sum_v);

        if self.refractory_state > 0 {
            self.refractory_state -= 1;
            self.v = self.e_l;
        }else{
            self.v += rk4(d_v, self.v, self.dt);
        }

        let mut spike = 0;
        let v = self.v;
        if self.v >= self.theta {
            self.v = self.e_l;
            self.refractory_state = self.refractory_period;
            spike = 1;
        }

        (spike, v)
    }

    pub fn update(&mut self, inputs: &Vec<Vec<u8>>, spike_history: &Vec<u8>, interval: usize) {
        // stdp learning

        const TAU_P: f64 = 0.005;
        const TAU_M: f64 = 0.005;
        const A_P: f64 = 0.5;
        const A_M: f64 = 0.05;
        const MIN_W: f64 = 0.0;

        let t_len = inputs[0].len();

        for t in 0..t_len {
            for i in 0..inputs.len() {

                if spike_history[t] == 1 {

                    // back
                    for j in 0..(interval as i64) {
                        let t_b = -j - 1;
                        if ((t as i64)+t_b) > 0 && inputs[i][((t as i64) +t_b) as usize] == 1 {
                            // update weight
                            self.weights[i] += A_P * (((t_b as f64)*self.dt) / TAU_P).exp();
                        }
                    }


                    // forward
                    for j in 0..interval {
                        let t_f = j + 1;
                        if (t+t_f) < t_len && inputs[i][t+t_f] == 1 {
                            // down weight
                            self.weights[i] -= A_M * ((-(t_f as f64)*self.dt) / TAU_M).exp();
                        }
                    }


                }

            }

        }

        let max_weight = self.weights.iter().fold(0.0/0.0, |m, v| v.max(m));
        for i in 0..inputs.len() {
            if self.weights[i] < MIN_W {
                self.weights[i] = MIN_W;
            }else{
                self.weights[i] /= max_weight;
            }
        }

    }
}


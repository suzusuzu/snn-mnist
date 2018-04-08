extern crate snn_mnist;
extern crate image;
extern crate rand;
extern crate rayon;

use rand::{Rng, SeedableRng, XorShiftRng};
use rayon::prelude::*;

use snn_mnist::mnist::{WIDTH, HEIGHT, NUM_PIXES, NUM_TRAIN};

fn main() {

    const EPOCH: usize = 10;
    const DT: f64 = 0.01;
    const NUM_UNIT: usize = 100;
    const RUN_TIME: f64 = 1.0;
    const V_INIT: f64 = -65.0;
    const THETA: f64 = -60.0;
    const REFACTORY_PERIOD: usize = 1;
    const MAX_FREQ: f64 = 60.0;
    const INTERVAL:usize = 10;

    let seed = 1;
    let x: u32 = 123456789;
    let y: u32 = 362436069;
    let z: u32 = 521288629;
    let w: u32 = seed;
    let seeds: [u32; 4] = [x, y, z, w];

    let mut rng = XorShiftRng::from_seed(seeds);


    // load MNIST
    let data = snn_mnist::mnist::train_data_load(NUM_TRAIN);

    // add neuron
    let mut layer = Vec::with_capacity(NUM_UNIT);
    for _ in 0..NUM_UNIT {
        let mut neuron = snn_mnist::neuron::Neuron::new(
            V_INIT, V_INIT, THETA, DT, NUM_PIXES, REFACTORY_PERIOD, &mut rng
        );
        layer.push(neuron);
    }

    // learning
    for e in 0..EPOCH {
        println!("EPOCH:{:2}", e);
        for n in 0..NUM_TRAIN {
            let mut img: Vec<Vec<u8>> = Vec::with_capacity(28);
            for i in 0..28 {
                let line = data[(i*WIDTH + n*NUM_PIXES)..((i+1)*WIDTH +n*NUM_PIXES)].to_vec();
                img.push(line);
            }
            let pot = snn_mnist::receptive_field::conv(&img, WIDTH, HEIGHT);
            let pot = snn_mnist::rate_coding::encode(&pot, MAX_FREQ, WIDTH, HEIGHT);

            let mut input_spikes = Vec::with_capacity(NUM_PIXES);

            for i in 0..WIDTH {
                for j in 0..HEIGHT {
                    let p = snn_mnist::poisson_spike::generate_spike(pot[i][j], DT, RUN_TIME, &mut rng);
                    input_spikes.push(p);
                }
            }

            let mut spike_historys = vec![Vec::new();NUM_UNIT];
            for t in 0..input_spikes[0].len() {
                let mut inputs = Vec::with_capacity(NUM_PIXES);
                for i in 0..NUM_PIXES {
                    inputs.push(input_spikes[i][t]);
                }
                let mut spikes = Vec::with_capacity(NUM_PIXES);
                for i in 0..NUM_UNIT {
                    let spike = layer[i].run(&inputs);
                    spikes.push(spike);
                }

                let spikes_sum = spikes.iter().sum::<u8>();
                if spikes_sum > 0 {
                    let spikes_tmp = spikes.clone();
                    let mut spikes_enu = spikes_tmp.iter().enumerate().collect::<Vec<(usize, &u8)>>();
                    rng.shuffle(&mut spikes_enu);
                    let &(spike_neuron, _) = spikes_enu.iter().max_by(|&&(_, a), &&(_, b)| a.cmp(&b)).unwrap();

                    // winner take all
                    for i in 0..NUM_UNIT {
                        if i != spike_neuron {
                            layer[i].v = V_INIT;
                            spikes[i] = 0;
                        }
                    }
                }

               for i in 0..NUM_UNIT {
                    spike_historys[i].push(spikes[i]);
                }
            }

            layer.par_iter_mut().enumerate()
                .for_each(|(i, neuron)|{
                    neuron.update(&input_spikes,  &spike_historys[i], INTERVAL);
                });

        }
    }

    // save weight
    for i in 0..NUM_UNIT {
        let max_weight = layer[i].weights.iter().fold(0.0/0.0, |m, v| v.max(m));
        let buf = layer[i].weights.iter().map(|&x| (x / max_weight * 255.0) as u8).collect::<Vec<u8>>();
        image::save_buffer(format!("image{}.png", i), &buf, WIDTH as u32, HEIGHT as u32, image::Gray(8)).unwrap()
    }
}
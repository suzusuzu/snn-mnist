extern crate image;
extern crate rand;
extern crate rayon;
extern crate snn_mnist;

use rand::{SeedableRng, XorShiftRng};
use rayon::prelude::*;

use snn_mnist::mnist::{HEIGHT, NUM_PIXES, NUM_TEST, NUM_TRAIN, WIDTH};

fn main() {
    const EPOCH: usize = 100;
    const DT: f64 = 0.025;
    const NUM_UNIT: usize = 900;
    const RUN_TIME: f64 = 1.0;
    const TEST_RUN_TIME: f64 = 3.0;
    const V_INIT: f64 = -75.0;
    const THETA: f64 = -60.0;
    const REFACTORY_PERIOD: usize = 1;
    const MAX_FREQ: f64 = 60.0;
    const INTERVAL: usize = 10;

    let seed = 1;
    let x: u32 = 123456789;
    let y: u32 = 362436069;
    let z: u32 = 521288629;
    let w: u32 = seed;
    let seeds: [u32; 4] = [x, y, z, w];

    let mut rng = XorShiftRng::from_seed(seeds);

    // load MNIST
    let data = snn_mnist::mnist::train_data_load(NUM_TRAIN);
    let labels = snn_mnist::mnist::train_label_load(NUM_TRAIN);
    let test_data = snn_mnist::mnist::test_data_load(NUM_TEST);
    let test_labels = snn_mnist::mnist::test_label_load(NUM_TEST);

    // add neuron
    let mut layer = Vec::with_capacity(NUM_UNIT);
    for _ in 0..NUM_UNIT {
        let neuron = snn_mnist::neuron::Neuron::new(
            V_INIT,
            V_INIT,
            THETA,
            DT,
            NUM_PIXES,
            REFACTORY_PERIOD,
            &mut rng,
        );
        layer.push(neuron);
    }

    // learning
    for e in 0..EPOCH {
        println!("epoch: {}", e);
        for n in 0..NUM_TRAIN {
            let mut img: Vec<Vec<u8>> = Vec::with_capacity(28);
            for i in 0..28 {
                let line =
                    data[(i * WIDTH + n * NUM_PIXES)..((i + 1) * WIDTH + n * NUM_PIXES)].to_vec();
                img.push(line);
            }
            let pot = snn_mnist::receptive_field::conv(&img, WIDTH, HEIGHT);
            let pot = snn_mnist::rate_coding::encode(&pot, MAX_FREQ, WIDTH, HEIGHT);

            let mut input_spikes = Vec::with_capacity(NUM_PIXES);

            for i in 0..WIDTH {
                for j in 0..HEIGHT {
                    let p =
                        snn_mnist::poisson_spike::generate_spike(pot[i][j], DT, RUN_TIME, &mut rng);
                    input_spikes.push(p);
                }
            }

            let mut spike_historys = vec![Vec::new(); NUM_UNIT];
            for t in 0..input_spikes[0].len() {
                let mut inputs = Vec::with_capacity(NUM_PIXES);
                for i in 0..NUM_PIXES {
                    inputs.push(input_spikes[i][t]);
                }
                let spikes = layer
                    .par_iter_mut()
                    .map(|neuron| neuron.run(&inputs))
                    .collect::<Vec<_>>();

                let mut fires = vec![0; NUM_UNIT];
                let spikes_sum = spikes.iter().map(|&(a, _)| a).sum::<u8>();
                if spikes_sum > 0 {
                    let spikes_tmp = spikes.iter().map(|&(_, b)| b).collect::<Vec<_>>();
                    let spikes_enu = spikes_tmp
                        .iter()
                        .enumerate()
                        .collect::<Vec<(usize, &f64)>>();
                    let &(spike_neuron, _) = spikes_enu
                        .iter()
                        .max_by(|&&(_, a), &&(_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    // winner take all
                    for i in 0..NUM_UNIT {
                        if i != spike_neuron {
                            layer[i].v = V_INIT;
                            fires[i] = 0;
                        } else {
                            fires[i] = 1;
                        }
                    }
                }

                for i in 0..NUM_UNIT {
                    spike_historys[i].push(fires[i]);
                }
            }

            layer.par_iter_mut().enumerate().for_each(|(i, neuron)| {
                neuron.update(&input_spikes, &spike_historys[i], INTERVAL);
            });
        }
    }

    // clustering
    let mut class_per_unit = vec![vec![0; 10]; NUM_UNIT];
    for n in 0..NUM_TRAIN {
        let mut img: Vec<Vec<u8>> = Vec::with_capacity(28);
        for i in 0..28 {
            let line =
                data[(i * WIDTH + n * NUM_PIXES)..((i + 1) * WIDTH + n * NUM_PIXES)].to_vec();
            img.push(line);
        }
        let pot = snn_mnist::receptive_field::conv(&img, WIDTH, HEIGHT);
        let pot = snn_mnist::rate_coding::encode(&pot, MAX_FREQ, WIDTH, HEIGHT);

        let mut input_spikes = Vec::with_capacity(NUM_PIXES);

        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                let p = snn_mnist::poisson_spike::generate_spike(
                    pot[i][j],
                    DT,
                    TEST_RUN_TIME,
                    &mut rng,
                );
                input_spikes.push(p);
            }
        }

        let mut spike_historys = vec![Vec::new(); NUM_UNIT];
        for t in 0..input_spikes[0].len() {
            let mut inputs = Vec::with_capacity(NUM_PIXES);
            for i in 0..NUM_PIXES {
                inputs.push(input_spikes[i][t]);
            }
            let spikes = layer
                .par_iter_mut()
                .map(|neuron| neuron.run(&inputs))
                .collect::<Vec<_>>();

            let mut fires = vec![0; NUM_UNIT];
            let spikes_sum = spikes.iter().map(|&(a, _)| a).sum::<u8>();
            if spikes_sum > 0 {
                let spikes_tmp = spikes.iter().map(|&(_, b)| b).collect::<Vec<_>>();
                let spikes_enu = spikes_tmp
                    .iter()
                    .enumerate()
                    .collect::<Vec<(usize, &f64)>>();
                let &(spike_neuron, _) = spikes_enu
                    .iter()
                    .max_by(|&&(_, a), &&(_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                // winner take all
                for i in 0..NUM_UNIT {
                    if i != spike_neuron {
                        layer[i].v = V_INIT;
                        fires[i] = 0;
                    } else {
                        fires[i] = 1;
                    }
                }
            }

            for i in 0..NUM_UNIT {
                spike_historys[i].push(fires[i]);
            }
        }
        let max_spike_unit = spike_historys
            .iter()
            .map(|spikes| spikes.iter().sum::<u8>())
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.cmp(&b))
            .unwrap()
            .0;
        let label = labels[n];
        class_per_unit[max_spike_unit][label as usize] += 1;
    }

    let class_per_unit = class_per_unit
        .iter()
        .map(|hist| {
            hist.iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.cmp(&b))
                .unwrap()
                .0
        })
        .collect::<Vec<usize>>();

    // test
    let mut num_correct = 0;
    for n in 0..NUM_TEST {
        let mut img: Vec<Vec<u8>> = Vec::with_capacity(28);
        for i in 0..28 {
            let line =
                test_data[(i * WIDTH + n * NUM_PIXES)..((i + 1) * WIDTH + n * NUM_PIXES)].to_vec();
            img.push(line);
        }
        let pot = snn_mnist::receptive_field::conv(&img, WIDTH, HEIGHT);
        let pot = snn_mnist::rate_coding::encode(&pot, MAX_FREQ, WIDTH, HEIGHT);

        let mut input_spikes = Vec::with_capacity(NUM_PIXES);

        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                let p = snn_mnist::poisson_spike::generate_spike(
                    pot[i][j],
                    DT,
                    TEST_RUN_TIME,
                    &mut rng,
                );
                input_spikes.push(p);
            }
        }

        let mut spike_historys = vec![Vec::new(); NUM_UNIT];
        for t in 0..input_spikes[0].len() {
            let mut inputs = Vec::with_capacity(NUM_PIXES);
            for i in 0..NUM_PIXES {
                inputs.push(input_spikes[i][t]);
            }
            let spikes = layer
                .par_iter_mut()
                .map(|neuron| neuron.run(&inputs))
                .collect::<Vec<_>>();

            let mut fires = vec![0; NUM_UNIT];
            let spikes_sum = spikes.iter().map(|&(a, _)| a).sum::<u8>();
            if spikes_sum > 0 {
                let spikes_tmp = spikes.iter().map(|&(_, b)| b).collect::<Vec<_>>();
                let spikes_enu = spikes_tmp
                    .iter()
                    .enumerate()
                    .collect::<Vec<(usize, &f64)>>();
                let &(spike_neuron, _) = spikes_enu
                    .iter()
                    .max_by(|&&(_, a), &&(_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                // winner take all
                for i in 0..NUM_UNIT {
                    if i != spike_neuron {
                        layer[i].v = V_INIT;
                        fires[i] = 0;
                    } else {
                        fires[i] = 1;
                    }
                }
            }

            for i in 0..NUM_UNIT {
                spike_historys[i].push(fires[i]);
            }
        }
        let max_spike_unit = spike_historys
            .iter()
            .map(|spikes| spikes.iter().sum::<u8>())
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.cmp(&b))
            .unwrap()
            .0;
        let label = test_labels[n];
        if label == (class_per_unit[max_spike_unit] as u8) {
            num_correct += 1;
        }
    }

    // save weight
    for i in 0..NUM_UNIT {
        let max_weight = layer[i].weights.iter().fold(0.0 / 0.0, |m, v| v.max(m));
        let buf = layer[i]
            .weights
            .iter()
            .map(|&x| (x / max_weight * 255.0) as u8)
            .collect::<Vec<u8>>();
        image::save_buffer(
            format!("image{}.png", i),
            &buf,
            WIDTH as u32,
            HEIGHT as u32,
            image::ColorType::L8,
        )
        .unwrap()
    }

    println!("acc: {}", (num_correct as f64) / (NUM_TEST as f64));
}

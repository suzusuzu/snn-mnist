extern crate snn_mnist;
extern crate image;
extern crate rand;

use rand::{thread_rng, Rng};

fn main() {

    let dt = 0.01;
    let num_unit = 100;
    // let num_unit = 100;
    let data = snn_mnist::mnist::train_data_load(6000);
    let label = snn_mnist::mnist::train_label_load(6000);
    // let data = snn_mnist::mnist::test_data_load(1000);
    // let label = snn_mnist::mnist::test_label_load(1000);

    let mut layer = Vec::with_capacity(num_unit);
    for _ in 0..num_unit {
        let mut neuron = snn_mnist::neuron::Neuron::new(
            -65.0, -65.0, -60.0, dt, 28*28, 1
        );
        layer.push(neuron);
    }

    for n in 0..(6000*30) {
        let nn = n;
        let n = n % 6000;
        // let n = n % 2 + 2;
        // let n = 0;
        let mut img: Vec<Vec<u8>> = Vec::with_capacity(28);
        for i in 0..28 {
            let line = data[(i*28+n*28*28)..((i+1)*28+n*28*28)].to_vec();
            img.push(line);
        }
        let pot = snn_mnist::receptive_field::conv(&img, 28, 28);
        let pot = snn_mnist::rate_coding::encode(&pot, 60.0, 28, 28);

        let mut input_spikes = Vec::with_capacity(28*28);

        for i in 0..28 {
            for j in 0..28 {
                let p = snn_mnist::poisson_spike::generate_spike(pot[i][j], dt, 0.3);
                input_spikes.push(p);
            }
        }

        let mut spike_historys = vec![Vec::new();num_unit];
        for t in 0..input_spikes[0].len() {
            let mut inputs = Vec::with_capacity(28*28);
            for i in 0..(28*28) {
                inputs.push(input_spikes[i][t]);
            }
            let mut spikes = Vec::with_capacity(28*28);
            for i in 0..num_unit {
                let spike = layer[i].run(&inputs);
                spikes.push(spike);
            }

            let spikes_sum = spikes.iter().sum::<u8>();
            if spikes_sum > 0 {
                let spikes_tmp = spikes.clone();
                let mut spikes_enu = spikes_tmp.iter().enumerate().collect::<Vec<(usize, &u8)>>();
                thread_rng().shuffle(&mut spikes_enu);
                let &(spike_neuron, _) = spikes_enu.iter().max_by(|&&(_, a), &&(_, b)| a.cmp(b)).unwrap();

                for i in 0..num_unit {
                    if i != spike_neuron {
                        layer[i].v = -65.0;
                        spikes[i] = 0;
                    }
                }
            }

            for i in 0..num_unit {
                spike_historys[i].push(spikes[i]);
            }
        }


        // winner take all


        /*
        let mut max_spike = -1000.0;
        let mut max_index = 0;

        for i in 0..num_unit {
            for t in 0..spike_historys[0].len() {
                if max_spike < spike_historys[i][t] {
                    max_spike = spike_historys[i][t];
                    max_index = i;
                }
            }
        }

        for i in 0..num_unit {
            for t in 0..spike_historys[0].len() {
                if i != max_index {
                    spike_historys[i][t] = -70.0;
                }
            }
        }
        println!("{}", max_index);
        */


        /*
        let mut spike_cnts = vec![0;num_unit];
        for i in 0..num_unit {
            for t in 0..spike_historys[0].len() {
                if spike_historys[i][t] > -60.0 {
                    spike_cnts[i] += 1;
                }
            }
        }
        // let (max_spike_neuron, cnt) = spike_cnts.iter().enumerate().max_by(|&(_, a), &(_, b)| a.cmp(b)).unwrap();
        let mut spike_cnts_enu = spike_cnts.iter().enumerate().collect::<Vec<(usize, &i32)>>();
        thread_rng().shuffle(&mut spike_cnts_enu);
        let &(max_spike_neuron, cnt) = spike_cnts_enu.iter().max_by(|&&(_, a), &&(_, b)| a.cmp(b)).unwrap();
        // println!("{}", cnt);

        for i in 0..num_unit {
            for t in 0..spike_historys[0].len() {
                if i != max_spike_neuron {
                    spike_historys[i][t] = -65.0;
                }
            }
        }
        */

        /*
        if *cnt != 0 {
            for i in 0..(28*28){
                if input_spikes[i].iter().sum::<u8>() == 0 {
                    layer[max_spike_neuron].weights[i] -= 0.001;
                    if layer[max_spike_neuron].weights[i] < 0.0 {
                        layer[max_spike_neuron].weights[i] = 0.0;
                    }
                }
            }
        }
        */


        /*
        for i in 0..(28*28){
            for t in 0..input_spikes[0].len() {
                if i != max_spike_neuron && input_spikes[i][t] == 0 {
                    layer[max_spike_neuron].weights[i] -= 0.0001;
                    if layer[max_spike_neuron].weights[i] < 0.0 {
                        layer[max_spike_neuron].weights[i] = 0.0;
                    }
                }
            }
        }
        */


        /*
        let mut winner_list = vec![0;num_unit];
        let t_len = spike_historys[0].len();
        for t in 0..spike_historys[0].len() {

            let mut max_spike = -1000.0;
            let mut max_index = 0;
            for i in 0..num_unit {
                // println!("{}", spike_historys[i][t]);
                if max_spike < spike_historys[i][t] {
                    max_spike = spike_historys[i][t];
                    max_index = i;
                }
            }

            if max_spike > -60.0 {
                for i in 0..num_unit {
                    if max_index != i {
                        spike_historys[i][t] = -70.0;
                    }
                }
            }

            // println!("{}", spike_historys[max_index][t]);
            if spike_historys[max_index][t] > -60.0 {
                winner_list[max_index] += 1;
            }
            // println!("{:?}", layer[max_index].weights);
        }
        */

        /*
        let (winner_index, cnt) = winner_list.iter().enumerate().max_by(|&(_, a), &(_, b)| a.cmp(b)).unwrap();

        if *cnt != 0 {
            for i in 0..(28*28){
                if input_spikes[i].iter().sum::<u8>() == 0 {
                    layer[winner_index].weights[i] -= 0.001;
                    if layer[winner_index].weights[i] < 0.0 {
                        layer[winner_index].weights[i] = 0.0;
                    }
                }
            }
        }
        */

        for i in 0..num_unit {
            layer[i].update(&input_spikes,  &spike_historys[i], 10);
        }

        // println!("{:?}", winner_list);
        // println!("{}", winner_index);
        // println!("{:?}", layer[1].weights.iter().sum::<f64>());
        /*
        if label[n] == 1 {
            zero_winners[winner_index] += 1;
        }*/

        let max_weight = layer[nn%num_unit].weights.iter().fold(0.0/0.0, |m, v| v.max(m));
        for i in 0..28 {
            for j in 0..28 {
                if layer[nn%num_unit].weights[j+i*28] > 0.0 {
                    print!("{:4}", (layer[nn%num_unit].weights[j+i*28] / max_weight * 255.0 ) as usize);
                } else {
                    print!("{:4}", 0 );
                }
            }
            println!("");
        }
        println!("");

    }

    for i in 0..num_unit {
        let max_weight = layer[i].weights.iter().fold(0.0/0.0, |m, v| v.max(m));
        let buf = layer[i].weights.iter().map(|&x| (x / max_weight * 255.0) as u8).collect::<Vec<u8>>();
        image::save_buffer(format!("image{}.png", i), &buf, 28, 28, image::Gray(8)).unwrap()
        // println!("{:?}", zero_winners);
    }

}
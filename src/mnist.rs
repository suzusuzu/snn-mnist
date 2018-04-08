use std::fs::File;
use std::io::prelude::*;

pub const WIDTH: usize = 28;
pub const HEIGHT: usize = 28;
pub const NUM_PIXES: usize = 784;
pub const NUM_TRAIN: usize = 6000;
pub const NUM_TEST: usize = 1000;

pub fn train_data_load(n: usize) -> Vec<u8> {
    let num_header = 16;
    let num_buf = n * 28 * 28 + num_header;
    let mut v = vec![0u8;num_buf];
    let mut file = File::open("train-images-idx3-ubyte").unwrap();
    file.read_exact(v.as_mut_slice()).unwrap();
    v[num_header..num_buf].to_vec()
}

pub fn train_label_load(n: usize) -> Vec<u8> {
    let num_header = 8;
    let num_buf = n + num_header;
    let mut v = vec![0u8;num_buf];
    let mut file = File::open("train-labels-idx1-ubyte").unwrap();
    file.read_exact(v.as_mut_slice()).unwrap();
    v[num_header..num_buf].to_vec()
}



pub fn test_data_load(n: usize) -> Vec<u8> {
    let num_header = 16;
    let num_buf = n * 28 * 28 + num_header;
    let mut v = vec![0u8;num_buf];
    let mut file = File::open("t10k-images-idx3-ubyte").unwrap();
    file.read_exact(v.as_mut_slice()).unwrap();
    v[num_header..num_buf].to_vec()
}

pub fn test_label_load(n: usize) -> Vec<u8> {
    let num_header = 8;
    let num_buf = n + num_header;
    let mut v = vec![0u8;num_buf];
    let mut file = File::open("t10k-labels-idx1-ubyte").unwrap();
    file.read_exact(v.as_mut_slice()).unwrap();
    v[num_header..num_buf].to_vec()
}


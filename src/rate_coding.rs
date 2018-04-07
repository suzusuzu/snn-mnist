pub fn encode(v: &Vec<Vec<f64>>, max_freq: f64, width: usize, height: usize) -> Vec<Vec<f64>> {

    let mut max = v[0][0];
    for i in 0..height {
        for j in 0..width {
            max = if max < v[i][j] { v[i][j] } else { max };
        }
    }

    let mut freqs = vec![vec![0.0; width]; height];

    for i in 0..height {
        for j in 0..width {
            if v[i][j] > 0.0 {
                freqs[i][j] = v[i][j] / max * max_freq;
            }else{
                freqs[i][j] = 0.0
            }
        }
    }

    freqs
}
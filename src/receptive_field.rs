pub fn conv(img: &Vec<Vec<u8>>, width: usize, height: usize) -> Vec<Vec<f64>> {
    let w0: f64 = 1.0;
    let w1: f64 = 0.7;
    let w2: f64 = 0.4;
    let w3: f64 = 0.1;
    let w4: f64 = -0.2;
    let w5: f64 = -0.5;
    let w: Vec<Vec<f64>> = vec![
        vec![w5, w4, w3, w4, w5],
        vec![w4, w2, w1, w2, w4],
        vec![w3, w1, w0, w1, w3],
        vec![w4, w2, w1, w2, w4],
        vec![w5, w4, w3, w4, w5],
    ];

    let mut pot = vec![vec![0.0; width]; height];
    for i in 0..height {
        for j in 0..width {
            let mut sum = 0.0;
            for k in 0..5 {
                for h in 0..5 {
                    if (i + k) >= 2 && (i + k - 2) < 28 && (j + h) >= 2 && (j + h - 2) < 28 {
                        sum += w[k][h] * (img[i + k - 2][j + h - 2] as f64);
                    }
                }
            }
            pot[i][j] = sum;
        }
    }

    pot
}

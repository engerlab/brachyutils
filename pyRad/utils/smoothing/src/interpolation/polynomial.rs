use dose::mc_dose::MCDose;

pub struct SmoothingPolynomial {
    pub b_000: f64,
    pub b_100: f64,
    pub b_010: f64,
    pub b_001: f64,
    pub b_110: f64,
    pub b_101: f64,
    pub b_011: f64,
    pub b_200: f64,
    pub b_020: f64,
    pub b_002: f64,
    pub origin: [i32; 3],
    pub win_i: i32,
    pub win_j: i32,
    pub win_k: i32,
    pub win_size: i32,
    pub num_dof: i32
}

impl SmoothingPolynomial {
    pub fn find_polynomial(dose: &MCDose, lin_voxel: i32, win_i: i32, win_j: i32, win_k: i32) -> SmoothingPolynomial {
        // Total number of voxels considered in the smoothing window
        let n_ijk = (2 * win_i + 1) * (2 * win_j + 1) * (2 * win_k + 1);

        // Explicitly convert integers to floating point values
        let fwin_i = win_i as f64;
        let fwin_j = win_j as f64;
        let fwin_k = win_k as f64;

        // Constants found in the polynomial coefficient equations.
        let g_i = fwin_i * (fwin_i + 1.0) / 3.0;
        let g_j = fwin_j * (fwin_j + 1.0) / 3.0;
        let g_k = fwin_k * (fwin_k + 1.0) / 3.0;

        let h_i = (3.0 * fwin_i * fwin_i + 3.0 * fwin_i - 1.0) / 5.0;
        let h_j = (3.0 * fwin_j * fwin_j + 3.0 * fwin_j - 1.0) / 5.0;
        let h_k = (3.0 * fwin_k * fwin_k + 3.0 * fwin_k - 1.0) / 5.0;

        let voxel = dose.vox_from_linear(lin_voxel);

        // Polynomial coefficients
        let mut b_000 = 0.0;

        let mut b_100 = 0.0;
        let mut b_010 = 0.0;
        let mut b_001 = 0.0;

        let mut b_110 = 0.0;
        let mut b_101 = 0.0;
        let mut b_011 = 0.0;

        let mut b_200 = 0.0;
        let mut b_020 = 0.0;
        let mut b_002 = 0.0;

        for n_k in -win_k..win_k+1 {
            for n_j in -win_j..win_j+1 {
                for n_i in -win_i..win_i+1 {
                    let (n_i_sq, n_j_sq, n_k_sq) = ((n_i * n_i) as f64, (n_j * n_j) as f64, (n_k * n_k) as f64);

                    let current_vox: [i32; 3] = [voxel[0] + n_i, voxel[1] + n_j, voxel[2] + n_k];
                    let lin_vox = dose.linear_from_vox(current_vox);

                    let dose_value = dose.values[lin_vox as usize];

                    let i_contrib = (n_i_sq - g_i) / (4.0 * g_i - 1.0);
                    let j_contrib = (n_j_sq - g_j) / (4.0 * g_j - 1.0);
                    let k_contrib = (n_k_sq - g_k) / (4.0 * g_k - 1.0);

                    b_000 += dose_value * (1.0 - 5.0 * (i_contrib + j_contrib + k_contrib));

                    if g_i > 0.0 {
                        b_100 += dose_value * n_i as f64 / g_i;
                        b_200 += dose_value * (g_i - n_i_sq) / (g_i * g_i - h_i * g_i);
                    }

                    if g_j > 0.0 {
                        b_010 += dose_value * n_j as f64 / g_j;
                        b_020 += dose_value * (g_j - n_j_sq) / (g_j * g_j - h_j * g_j);
                    }

                    if g_k > 0.0 {
                        b_001 += dose_value * n_k as f64 / g_k;
                        b_002 += dose_value * (g_k - n_k_sq) / (g_k * g_k - h_k * g_k);
                    }

                    if g_i > 0.0 && g_j > 0.0 {
                        b_110 += dose_value * n_i as f64 * n_j as f64 / (g_i * g_j);
                    }

                    if g_i > 0.0 && g_k > 0.0 {
                        b_101 += dose_value * n_i as f64 * n_k as f64 / (g_i * g_k);
                    }

                    if g_j > 0.0 && g_k > 0.0 {
                        b_011 += dose_value * n_j as f64 * n_k as f64 / (g_j * g_k);
                    }
                }
            }
        }

        b_000 /= n_ijk as f64;

        b_100 /= n_ijk as f64;
        b_010 /= n_ijk as f64;
        b_001 /= n_ijk as f64;

        b_110 /= n_ijk as f64;
        b_101 /= n_ijk as f64;
        b_011 /= n_ijk as f64;

        b_200 /= n_ijk as f64;
        b_020 /= n_ijk as f64;
        b_002 /= n_ijk as f64;

        let num_dof;
        if (win_i == 0 && win_j == 0) ||
           (win_i == 0 && win_k == 0) ||
           (win_j == 0 && win_k == 0) {
            num_dof = 3;
        } else if win_i > 0 && win_j > 0 && win_k > 0 {
            num_dof = 10;
        } else {
            num_dof = 6;
        }

        SmoothingPolynomial {
            b_000: b_000,

            b_100: b_100, b_010: b_010, b_001: b_001,

            b_110: b_110, b_101: b_101, b_011: b_011,

            b_200: b_200, b_020: b_020, b_002: b_002,

            origin: voxel,
            win_i: win_i,
            win_j: win_j,
            win_k: win_k,
            win_size: n_ijk,
            num_dof: num_dof
        }
    }

    pub fn eval_from_dist(&self, dist_i: i32, dist_j: i32, dist_k: i32) -> f64 {
        // Must explicitly convert integers to floats
        let _i = dist_i as f64;
        let _j = dist_j as f64;
        let _k = dist_k as f64;

        self.b_000 +
        (self.b_100 * _i) + (self.b_010 * _j) + (self.b_001 * _k) +
        (self.b_110 * _i * _j) + (self.b_101 * _i * _k) + (self.b_011 * _j * _k) +
        (self.b_200 * (_i * _i)) + (self.b_020 * (_j * _j)) + (self.b_002 * (_k * _k))
    }

    pub fn eval_from_vox(&self, vox: [i32; 3]) -> f64 {
        // Must explicitly convert integers to floats
        let (_i, _j, _k) = ((vox[0] - self.origin[0]),
                              (vox[1] - self.origin[1]),
                              (vox[2] - self.origin[2]));

        self.eval_from_dist(_i, _j, _k)
    }

    pub fn print_coefficients(&self) {
        println!("b_000 = {:e}, b_100 = {:e}, b_010 = {:e}, b_001 = {:e}, b_110 = {:e}, b_101 = {:e}, b_011 = {:e}, b_200 = {:e}, b_020 = {:e}, b_002 = {:e}", self.b_000, self.b_100, self.b_010, self.b_001, self.b_110, self.b_101, self.b_011, self.b_200, self.b_020, self.b_002);
    }

    pub fn calculate_chisq(&self, dose: &MCDose) -> f64 {
        let mut chisq = 0.0;
        let mut avg_uncert = 0.0;
        let mut num_avg = 0;

        for n_k in -self.win_k..self.win_k+1 {
            for n_j in -self.win_j..self.win_j+1 {
                for n_i in -self.win_i..self.win_i+1 {
                    let current_vox: [i32; 3] = [self.origin[0] + n_i, self.origin[1] + n_j, self.origin[2] + n_k];
                    let lin_vox = dose.linear_from_vox(current_vox);

                    let dose_value = dose.values[lin_vox as usize];

                    if dose_value > 0.0 {
                        let p_ijk = self.eval_from_dist(n_i, n_j, n_k);

                        // Uncertainties are stored relative sigma to the dose value.
                        // we want the absolute uncertainty value.
                        //let uncert_value = dose.uncertainties[lin_vox as usize] * dose_value;
                        let uncert = dose.uncertainties[lin_vox as usize] * dose_value;
                        avg_uncert += uncert * uncert;
                        num_avg += 1;

                        let chi = p_ijk - dose_value;
                        chisq += chi * chi;
                    }
                }
            }
        }

        avg_uncert /= num_avg as f64;
        chisq / avg_uncert
    }

    pub fn chisq_test(&self, dose: &MCDose, max_chisq: f64) -> bool {
        let chisq = self.calculate_chisq(dose);

        let test = chisq / (self.win_size - self.num_dof) as f64;

        test <= max_chisq
    }
}

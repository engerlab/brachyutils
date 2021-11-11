use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;

use interpolation::polynomial::SmoothingPolynomial;

pub struct MCDose {
    pub num_voxels: [i32; 3],
    pub spacing: [f64; 3],
    pub topleft: [f64; 3],
    pub values: Vec<f64>,
    pub uncertainties: Vec<f64>
}

impl MCDose {
    pub fn from_file(file_path: &Path) -> MCDose {
        println!("Opening {}", file_path.display());
        let file = match File::open(&file_path) {
            Err(why) => panic!("could not open {}: {}", file_path.display(), why.description()),
            Ok(file) => file,
        };

        let mut reader = BufReader::new(file);
        let mut dose_line = String::new();

        // First line is number of voxels
        reader.read_line(&mut dose_line).unwrap();
        let voxels: Vec<i32> = dose_line.split_whitespace().map(|x| x.parse::<i32>().unwrap()).collect();
        dose_line.clear();

        // Coordinates of every voxel in the x direction
        reader.read_line(&mut dose_line).unwrap();
        let x_voxels: Vec<f64> = dose_line.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect();
        dose_line.clear();

        // Coordinates of every voxel in the y direction
        reader.read_line(&mut dose_line).unwrap();
        let y_voxels: Vec<f64> = dose_line.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect();
        dose_line.clear();

        // Coordinates of every voxel in the z direction
        reader.read_line(&mut dose_line).unwrap();
        let z_voxels: Vec<f64> = dose_line.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect();
        dose_line.clear();

        // Dose values
        reader.read_line(&mut dose_line).unwrap();
        let dose_values: Vec<f64> = dose_line.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect();
        dose_line.clear();

        // Uncertainty values
        reader.read_line(&mut dose_line).unwrap();
        let uncert_values: Vec<f64> = dose_line.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect();
        dose_line.clear();

        // There should be one dose value and one uncertainty value per voxel
        let total_voxels = voxels[0] * voxels[1] * voxels[2];
        assert_eq!(dose_values.len(), total_voxels as usize);
        assert_eq!(uncert_values.len(), total_voxels as usize);

        // Construct the dose struct
        let topleft = [x_voxels[0], y_voxels[0], z_voxels[0]];

        let spacing = [x_voxels[1] - x_voxels[0],
                       y_voxels[1] - y_voxels[0],
                       z_voxels[1] - z_voxels[0]];

        let num_voxels = [voxels[0], voxels[1], voxels[2]];

        MCDose {
            num_voxels: num_voxels,
            spacing: spacing,
            topleft: topleft,
            values: dose_values,
            uncertainties: uncert_values
        }
    }

    pub fn write_3ddose(&self, filename: String) {
        let path = Path::new(&filename);
        println!("Writing {}", path.display());

        let display = path.display();

        // Open a file in write-only mode, returns `io::Result<File>`
        let mut file = match File::create(&path) {
            Err(why) => panic!("couldn't create {}: {}",
                               display,
                               why.description()),
            Ok(file) => file,
        };

        write!(&mut file, "{} {} {}\n", self.num_voxels[0], self.num_voxels[1], self.num_voxels[2]).unwrap();

        let x_voxels: Vec<f64> = (0..self.num_voxels[0]+1).map(|x| x as f64 * self.spacing[0] + self.topleft[0]).collect();
        let y_voxels: Vec<f64> = (0..self.num_voxels[1]+1).map(|x| x as f64 * self.spacing[1] + self.topleft[1]).collect();
        let z_voxels: Vec<f64> = (0..self.num_voxels[2]+1).map(|x| x as f64 * self.spacing[2] + self.topleft[2]).collect();

        let string_x_voxels: Vec<String> = x_voxels.iter().map(|x| format!("{:.4}", x)).collect();
        let string_y_voxels: Vec<String> = y_voxels.iter().map(|x| format!("{:.4}", x)).collect();
        let string_z_voxels: Vec<String> = z_voxels.iter().map(|x| format!("{:.4}", x)).collect();

        write!(&mut file, "{}\n", string_x_voxels.join(" ")).unwrap();
        write!(&mut file, "{}\n", string_y_voxels.join(" ")).unwrap();
        write!(&mut file, "{}\n", string_z_voxels.join(" ")).unwrap();

        let string_doses: Vec<String> = self.values.iter().map(|x| format!("{:e}", x)).collect();
        write!(&mut file, "{}\n", string_doses.join(" ")).unwrap();

        let string_uncerts: Vec<String> = self.uncertainties.iter().map(|x| format!("{:e}", x)).collect();
        write!(&mut file, "{}\n", string_uncerts.join(" ")).unwrap();

    }

    pub fn vox_from_linear(&self, linear_vox: i32) -> [i32; 3] {
        let mut voxel = [0; 3];
        let voxel_xy = self.num_voxels[0] * self.num_voxels[1];
        voxel[2] = linear_vox / voxel_xy;
        voxel[1] = (linear_vox - (voxel[2] * voxel_xy)) / self.num_voxels[0];
        voxel[0] = linear_vox - (voxel[2] * voxel_xy) - (voxel[1] * self.num_voxels[0]);

        voxel
    }

    pub fn linear_from_vox(&self, vox: [i32; 3]) -> i32 {
        //let lin = vox[2] * self.num_voxels[0] * self.num_voxels[1] + vox[1] * self.num_voxels[0] + vox[0];
        vox[2] * self.num_voxels[0] * self.num_voxels[1] + vox[1] * self.num_voxels[0] + vox[0]
    }

    pub fn smooth(&self) -> MCDose {
        const MAX_CHISQ: f64 = 1.0;
        const MIN_WINDOW: i32 = 4;

        let total_voxels = self.num_voxels[0] * self.num_voxels[1] * self.num_voxels[2];

        let mut smoothed_dose: Vec<f64> = Vec::with_capacity(total_voxels as usize);

        let mut smoothed_voxels = 0;
        let mut processed_voxels = 0;

        for linear_vox in 0..total_voxels {
            let dose_value = self.values[linear_vox as usize];

            let mut max_wi = 0;
            let mut max_wj = 0;
            let mut max_wk = 0;
            let mut max_nijk = 0;

            if dose_value > 0.0 {
                processed_voxels += 1;

                // Do 1D searches for maximum window size for each dimension before
                // doing the general 3D search.
                let win_i = self.local_search_1d(linear_vox, 0);
                let win_j = self.local_search_1d(linear_vox, 1);
                let win_k = self.local_search_1d(linear_vox, 2);

                // 3D maximum window size search... basically identify the largest
                // window size that has chisq <= MAX_CHISQ.
                for w_k in (0..win_k+1).rev() {
                    for w_j in (0..win_j+1).rev() {
                        for w_i in (0..win_i+1).rev() {
                            let l_nijk = (2 * w_i + 1) * (2 * w_j + 1) * (2 * w_k + 1);
                            // If the window size is lower than the maximum acceptable window size,
                            // then break out of this part of the loop because we're iterating
                            // in descending order and the window size will only get smaller.
                            if l_nijk >= MIN_WINDOW && l_nijk > max_nijk {
                                let poly = SmoothingPolynomial::find_polynomial(self, linear_vox, w_i, w_j, w_k);

                                if poly.chisq_test(self, MAX_CHISQ) {
                                    max_wi = w_i;
                                    max_wj = w_j;
                                    max_wk = w_k;
                                    max_nijk = l_nijk;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
            }

            if max_nijk >= MIN_WINDOW {
                let poly = SmoothingPolynomial::find_polynomial(self, linear_vox, max_wi, max_wj, max_wk);
                smoothed_dose.push(poly.b_000);
                smoothed_voxels += 1;
            } else {
                smoothed_dose.push(self.values[linear_vox as usize]);
            }
        }

        println!("Fraction of smoothed voxels: {}", smoothed_voxels as f64 / processed_voxels as f64);

        // Sanity test to make sure that the smoothed dose has same number of voxels
        // as the unsmoothed dose.
        assert_eq!(smoothed_dose.len(), total_voxels as usize);

        MCDose {
            num_voxels: self.num_voxels,
            spacing: self.spacing,
            topleft: self.topleft,
            values: smoothed_dose,
            uncertainties: self.uncertainties.clone()
        }
    }

    fn local_search_1d(&self, linear_vox: i32, dim: i32) -> i32 {
        // Find largest window size in one dimension.

        const MAX_NW: i32 = 3;

        // In 1D, there are 3 free parameters for 2nd degree polynomial.
        // Need at least 4 data points to get a fit, and the data points are
        // given as (2 * nw + 1) so MIN_NW = 2.
        const MIN_NW: i32 = 2;
        const MAX_CHISQ: f64 = 1.0;

        let mut local_max = MAX_NW;

        let vox = self.vox_from_linear(linear_vox);

        // Check if we're close to a boundary
        if vox[dim as usize] - local_max < 0 {
            local_max = vox[dim as usize];
        } else if vox[dim as usize] + local_max >= self.num_voxels[dim as usize] {
            local_max = self.num_voxels[dim as usize] - vox[dim as usize] - 1;
        }

        while local_max >= MIN_NW {
            let poly: SmoothingPolynomial;
            if dim == 0 {
                poly = SmoothingPolynomial::find_polynomial(self, linear_vox, local_max, 0, 0);
            } else if dim == 1 {
                poly = SmoothingPolynomial::find_polynomial(self, linear_vox, 0, local_max, 0);
            } else {
                poly = SmoothingPolynomial::find_polynomial(self, linear_vox, 0, 0, local_max);
            }

            if poly.chisq_test(self, MAX_CHISQ) {
                return local_max;
            } else {
                local_max -= 1;
            }
        }

        local_max
    }
}

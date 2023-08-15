use std::ops::Index;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use image::{self, DynamicImage, ImageBuffer};
use image::{GenericImage, GenericImageView, Rgba};

fn sqaure_error_rgba(pixel_1: Rgba<u8>, pixel_2: Rgba<u8>) -> f64 {
    let mut square_error = 0.0;
    for i in 0..3 {
        let color_1 = pixel_1.index(i);
        let color_2 = pixel_2.index(i);
        square_error += f64::from(i32::pow(*color_1 as i32 - *color_2 as i32, 2));
        // println!("{}", color_1);
    }
    square_error
}

fn calculate_pixel_energy(img: &DynamicImage, x: u32, y: u32) -> f64 {
    let mut sqaure_error = 0.0;
    let pixel = img.get_pixel(x, y);
    // println!("==========> {:?}", pixel);
    for j in -1..2 {
        if (y as i32) + j < 0 {
            continue;
        }
        if ((y as i32) + j) as u32 >= img.dimensions().1 {
            continue;
        }
        for i in -1..2 {
            if (x as i32) + i < 0 {
                continue;
            }
            if ((x as i32) + i) as u32 >= img.dimensions().0 {
                continue;
            }
            if i == 0 && j == 0 {
                continue;
            }
            let neighbour_x = ((x as i32) + i) as u32;
            let neighbour_y = ((y as i32) + j) as u32;
            let neighbour = img.get_pixel(neighbour_x, neighbour_y);
            // println!("======= {:?}", pixel);
            sqaure_error += sqaure_error_rgba(pixel, neighbour);
        }
    }
    return sqaure_error;
}

fn fill_energy_matrix_async(img: &DynamicImage, dimensions: (u32, u32)) -> Vec<Vec<f64>> {
    let mut energy_matrix: Vec<Vec<f64>> =
        vec![vec![0.0; dimensions.0 as usize]; dimensions.1 as usize];
    let (tx, rx) = mpsc::channel();
    for y in 0..dimensions.1 {
        let tx_clone = tx.clone();
        let img_clone = img.clone();
        thread::spawn(move || {
            let mut energy_row: Vec<f64> = vec![0.0; dimensions.0 as usize];
            for x in 0..dimensions.0 {
                energy_row[x as usize] = calculate_pixel_energy(&img_clone, x, y);
            }
            tx_clone.send((y, energy_row)).unwrap();
        });
    }
    drop(tx);
    for (y, energy_row) in rx {
        energy_matrix[y as usize] = energy_row;
    }
    energy_matrix
}

fn fill_energy_matrix_async_other(img: &DynamicImage, dimensions: (u32, u32)) -> Vec<Vec<f64>> {
    let mut energy_matrix: Vec<Vec<f64>> =
        vec![vec![0.0; dimensions.0 as usize]; dimensions.1 as usize];
    let (tx, rx) = mpsc::channel();
    let img_clone = Arc::new(Mutex::new(img.clone()));
    for y in 0..dimensions.1 {
        let tx_clone = tx.clone();
        let img_clone = Arc::clone(&img_clone);
        thread::spawn(move || {
            let mut energy_row: Vec<f64> = vec![0.0; dimensions.0 as usize];
            let thread_img = img_clone.lock().unwrap();
            for x in 0..dimensions.0 {
                energy_row[x as usize] = calculate_pixel_energy(&*thread_img, x, y);
            }
            tx_clone.send((y, energy_row)).unwrap();
            // println!("finished: {y}");
        });
    }
    drop(tx);
    for (y, energy_row) in rx {
        energy_matrix[y as usize] = energy_row;
    }
    energy_matrix
}

fn update_energy_matrix(
    energy_matrix: &Vec<Vec<f64>>,
    seam: &Vec<(u32, u32)>,
    dimensions: (u32, u32),
) -> Vec<Vec<f64>> {
    let mut new_energy_matrix: Vec<Vec<f64>> = vec![vec![]; dimensions.1 as usize];
    for (x, y) in seam {
        let mut new_energy_row = energy_matrix[*y as usize][..*x as usize].to_vec();
        let mut new_energy_row_second_half =
            energy_matrix[*y as usize][(x + 1) as usize..].to_vec();

        new_energy_row.append(&mut new_energy_row_second_half);
        new_energy_matrix[*y as usize] = new_energy_row;
    }
    new_energy_matrix
}

fn retrace_seam(p_matrix: &Vec<Vec<i8>>, k: usize, y_bound: u32) -> Vec<(u32, u32)> {
    let mut seam: Vec<(u32, u32)> = vec![(0, 0); y_bound as usize];
    let mut j = k as i32;
    for i in 0..y_bound {
        seam[(y_bound - (i as u32) - 1) as usize] = (j as u32, y_bound - (i as u32) - 1);
        j += p_matrix[(y_bound - (i as u32) - 1) as usize][j as usize] as i32;
    }
    seam
}

fn find_best_seam(
    energy_matrix: &Vec<Vec<f64>>,
    dimensions: (u32, u32),
    reduction: u32,
) -> Vec<Vec<(u32, u32)>> {
    let mut optimality_matrix: Vec<Vec<f64>> =
        vec![vec![0.0; dimensions.0 as usize]; dimensions.1 as usize];
    let mut p_matrix: Vec<Vec<i8>> = vec![vec![0; dimensions.0 as usize]; dimensions.1 as usize];
    optimality_matrix[0] = energy_matrix[0].clone();
    p_matrix[0] = vec![0; dimensions.0 as usize];
    // println!("{:?}", dimensions);
    for i in 1..(dimensions.1 as usize) {
        // let optimality_row: Vec<f64> = vec![0.0; dimensions.0 as usize];
        // let p_row: Vec<i8> = vec![0; dimensions.0 as usize];
        for j in 0..(dimensions.0 as usize) {
            optimality_matrix[i][j] = optimality_matrix[(i as i32 - 1) as usize][j];
            p_matrix[i][j] = 0;
            if j > 1 {
                if optimality_matrix[(i as i32 - 1) as usize][(j as i32 - 1) as usize]
                    < optimality_matrix[i][j]
                {
                    optimality_matrix[i][j] =
                        optimality_matrix[(i as i32 - 1) as usize][(j as i32 - 1) as usize];
                    p_matrix[i][j] = -1;
                }
            }
            if j < (dimensions.0 as i32 - 2) as usize {
                if optimality_matrix[(i as i32 - 1) as usize][(j as i32 + 1) as usize]
                    < optimality_matrix[i][j]
                {
                    optimality_matrix[i][j] =
                        optimality_matrix[(i as i32 - 1) as usize][(j as i32 + 1) as usize];
                    p_matrix[i][j] = 1;
                }
            }
            optimality_matrix[i][j] = optimality_matrix[i][j] + energy_matrix[i][j];

            if p_matrix[(j as i32 - 1) as usize][(i as i32 + p_matrix[i][j] as i32) as usize]
                == i8::MIN
            {
                p_matrix[i][j] = i8::MIN;
            }
            match p_matrix[i][j] {
                0 => {
                    if j <= 1 {
                        break;
                    }
                    if p_matrix[i][(j as i32 - 1) as usize] != 1 {
                        break;
                    }
                    if optimality_matrix[i][j] < optimality_matrix[i][(j as i32 - 1) as usize] {
                        p_matrix[i][j] = i8::MIN;
                    } else {
                        p_matrix[i][(j as i32 - 1) as usize] = i8::MIN;
                    }
                }
                -1 => {
                    if j <= 1 {
                        break;
                    }
                    if p_matrix[i][(j as i32 - 1) as usize] == 0 {
                        if optimality_matrix[i][j] < optimality_matrix[i][(j as i32 - 1) as usize] {
                            p_matrix[i][j] = i8::MIN;
                        } else {
                            p_matrix[i][(j as i32 - 1) as usize] = i8::MIN;
                        }
                    }

                    if j <= 2 {
                        break;
                    }
                    if p_matrix[i][(j as i32 - 2) as usize] != 1 {
                        break;
                    }
                    if optimality_matrix[i][j] < optimality_matrix[i][(j as i32 - 2) as usize] {
                        p_matrix[i][j] = i8::MIN;
                    } else {
                        p_matrix[i][(j as i32 - 2) as usize] = i8::MIN;
                    }
                }
                _ => (),
            }
        }
    }

    let mut valid_idxs: Vec<usize> = vec![];

    for (idx, pointer) in p_matrix.last().unwrap().iter().enumerate() {
        if *pointer != i8::MIN {
            valid_idxs.push(idx);
        }
    }

    let mut seams: Vec<Vec<(u32, u32)>> = vec![];

    for _ in 0..reduction {
        let mut current_max = f64::MAX;
        let mut current_idx_of_idx = 0;
        let mut current_idx = 0;
        for (idx_of_idx, idx) in valid_idxs.iter().enumerate() {
            current_max = f64::max(optimality_matrix.last().unwrap()[*idx], current_max);
            current_idx_of_idx = idx_of_idx;
            current_idx = *idx;
        }
        if current_max == f64::MAX {
            panic!("not enough seams");
        }
        let seam = retrace_seam(&p_matrix, current_idx, dimensions.1);
        seams.push(seam);
        valid_idxs.remove(current_idx_of_idx);
    }
    // println!("{}", k);
    seams
}

fn fill_new_image(
    old_img: &DynamicImage,
    seam: &Vec<(u32, u32)>,
    dimensions: (u32, u32),
) -> DynamicImage {
    let mut new_img = DynamicImage::new_rgba8((dimensions.0 as i32 - 1) as u32, dimensions.1);
    for y in 0..dimensions.1 {
        let mut seam_flag = 0;
        for x in 0..(dimensions.0 as i32 - 1) {
            if seam[y as usize].0 == x as u32 {
                seam_flag += 1;
            }
            new_img.put_pixel(x as u32, y, old_img.get_pixel((x + seam_flag) as u32, y));
        }
    }
    new_img
}

fn fill_new_image2(
    old_img: &DynamicImage,
    mask: &mut Vec<Vec<(u32, u32)>>,
    dimensions: (u32, u32),
) -> DynamicImage {
    let mut new_img = DynamicImage::new_rgba8(dimensions.0, dimensions.1);
    for y in 0..dimensions.1 {
        for x in 0..dimensions.0 {
            new_img.put_pixel(
                x,
                y,
                old_img.get_pixel(
                    mask[y as usize][x as usize].0,
                    mask[y as usize][x as usize].1,
                ),
            );
        }
    }
    new_img
}

fn update_image_mask(mask: &mut Vec<Vec<(u32, u32)>>, seams: Vec<Vec<(u32, u32)>>) {
    for seam in seams {
        for (x, y) in seam {
            let mut first_half = mask[y as usize][0..x as usize].to_vec();
            let mut second_half = if x < mask[0].len() as u32 {
                mask[y as usize][((x + 1) as usize)..].to_vec()
            } else {
                vec![]
            };
            first_half.append(&mut second_half);
            mask[y as usize] = first_half;
        }
    }
}

fn fill_image_mask(dimensions: (u32, u32)) -> Vec<Vec<(u32, u32)>> {
    let mut mask = vec![vec![(0u32, 0u32); dimensions.0 as usize]; dimensions.1 as usize];
    for y in 0..dimensions.1 {
        for x in 0..dimensions.0 {
            mask[y as usize][x as usize] = (x, y);
        }
    }
    mask
}

// fn main() {
//     let img = image::open(
//         "/home/saran/testProjects/rust/seam-carving-rs/8916568062e761d403a6f0673b13943c.jpeg",
//     )
//     .unwrap();

//     let dimensions = img.dimensions();
//     let mut x = dimensions.0;
//     let mut y = dimensions.1;
//     // let mut out: image::RgbaImage = image::ImageBuffer::new(width, height);
//     // let mut i = 0;
//     let reduction = 20u32;

//     let now = Instant::now();
//     println!("start");
//     let mut energy_matrix = fill_energy_matrix_async(&img, dimensions);
//     println!("Calculated energy matrix");
//     println!("{:?}", now.elapsed());
//     // let mut seam = find_best_seam(&energy_matrix, dimensions);
//     // println!("Calculated best seam");
//     // let mut new_img = fill_new_image(&img, &seam, dimensions);
//     // println!("Filled new image");
//     let mut seams = vec![vec![(0u32, 0u32); dimensions.1 as usize]; reduction as usize];
//     let mut img_mask = fill_image_mask(dimensions);
//     for i in 0..reduction {
//         let now = Instant::now();
//         println!("{i}");
//         let seam = find_best_seam(&energy_matrix, (x, y));
//         println!("Calculated best seam");
//         println!("{:?}", now.elapsed());
//         let now = Instant::now();
//         energy_matrix = update_energy_matrix(&energy_matrix, &seam, (x, y));
//         println!("Calculated energy matrix");
//         println!("{:?}", now.elapsed());
//         println!("Length of energy matrix: {}", energy_matrix[0].len());
//         seams[i as usize] = seam;
//         x -= 1;
//     }
//     let now_1 = Instant::now();
//     update_image_mask(&mut img_mask, seams);
//     println!("Updates image mask");
//     println!("{:?}", now_1.elapsed());
//     let now_2 = Instant::now();
//     let new_img = fill_new_image2(
//         &img,
//         &mut img_mask,
//         (dimensions.0 - reduction, dimensions.1),
//     );
//     println!("Filled new image");
//     println!("{:?}", now_2.elapsed());

//     println!("Total time");
//     println!("{:?}", now.elapsed());
//     new_img.save("new_img.png").unwrap();
//     // println!("{:?}", seam);
//     // for pixel in out.enumerate_pixels_mut() {
//     //     *pixel.2 = image::Rgba([
//     //         ((pixel.0 + pixel.1) / 13) as u8,
//     //         ((pixel.1 + pixel.1) / 13) as u8,
//     //         ((pixel.0 + pixel.1) / 13) as u8,
//     //         255,
//     //     ]);
//     // println!("----------> {:?}", img.get_pixel(pixel.0, pixel.1));
//     // println!("{} {}", i, calculate_pixel_energy(&img, pixel.0, pixel.1));
//     // if i > 1000 {
//     //     break;
//     // }
//     // i += 1;
//     // }
//     // out.save("out.png").unwrap();
//     // img.save("test.png").unwrap();
// }

fn main() {
    let img = image::open(
        "/home/saran/testProjects/rust/seam-carving-rs/8916568062e761d403a6f0673b13943c.jpeg",
    )
    .unwrap();

    let dimensions = img.dimensions();
    let mut x = dimensions.0;
    let mut y = dimensions.1;
    // let mut out: image::RgbaImage = image::ImageBuffer::new(width, height);
    // let mut i = 0;
    let reduction = 20u32;

    let now = Instant::now();
    println!("start");
    let mut energy_matrix = fill_energy_matrix_async(&img, dimensions);
    println!("Calculated energy matrix");
    println!("{:?}", now.elapsed());
    // let mut seam = find_best_seam(&energy_matrix, dimensions);
    // println!("Calculated best seam");
    // let mut new_img = fill_new_image(&img, &seam, dimensions);
    // println!("Filled new image");
    let mut seams = find_best_seam(&energy_matrix, dimensions, reduction);
    let mut img_mask = fill_image_mask(dimensions);
    let now_1 = Instant::now();
    update_image_mask(&mut img_mask, seams);
    println!("Updates image mask");
    println!("{:?}", now_1.elapsed());
    let now_2 = Instant::now();
    let new_img = fill_new_image2(
        &img,
        &mut img_mask,
        (dimensions.0 - reduction, dimensions.1),
    );
    println!("Filled new image");
    println!("{:?}", now_2.elapsed());

    println!("Total time");
    println!("{:?}", now.elapsed());
    new_img.save("new_img.png").unwrap();
    // println!("{:?}", seam);
    // for pixel in out.enumerate_pixels_mut() {
    //     *pixel.2 = image::Rgba([
    //         ((pixel.0 + pixel.1) / 13) as u8,
    //         ((pixel.1 + pixel.1) / 13) as u8,
    //         ((pixel.0 + pixel.1) / 13) as u8,
    //         255,
    //     ]);
    // println!("----------> {:?}", img.get_pixel(pixel.0, pixel.1));
    // println!("{} {}", i, calculate_pixel_energy(&img, pixel.0, pixel.1));
    // if i > 1000 {
    //     break;
    // }
    // i += 1;
    // }
    // out.save("out.png").unwrap();
    // img.save("test.png").unwrap();
}

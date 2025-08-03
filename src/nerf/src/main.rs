mod camera;
mod hittable;
mod material;
mod ray;
mod sphere;

use crate::camera::Camera;
use crate::hittable::{Hitable, HitableList};
use crate::material::*;
use crate::sphere::Sphere;
use nalgebra::Vector3;
use rand::prelude::*;
use rayon::prelude::*;
use std::f32;
use std::fs::File;
use std::io::Write;

fn random_scene() -> HitableList {
    let mut rng = SmallRng::seed_from_u64(42);
    let origin = Vector3::new(4.0, 0.2, 0.0);
    let mut world = HitableList::default();
    world.push(Sphere::new(
        Vector3::new(0.0, -1000.0, 0.0),
        1000.0,
        Lambertian::new(Vector3::new(0.5, 0.5, 0.5)),
    ));
    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random_range(0.0..1.0);
            let center = Vector3::new(
                a as f32 + 0.9 * rng.random_range(0.0..1.0),
                0.2,
                b as f32 + 0.9 * rng.random_range(0.0..1.0),
            );
            if (center - origin).magnitude() > 0.9 {
                if choose_material < 0.8 {
                    // diffuse
                    world.push(Sphere::new(
                        center,
                        0.2,
                        Lambertian::new(Vector3::new(
                            rng.random_range(0.0..1.0) * rng.random_range(0.0..1.0),
                            rng.random_range(0.0..1.0) * rng.random_range(0.0..1.0),
                            rng.random_range(0.0..1.0) * rng.random_range(0.0..1.0),
                        )),
                    ));
                } else if choose_material < 0.95 {
                    // metal
                    world.push(Sphere::new(
                        center,
                        0.2,
                        Metal::new(
                            Vector3::new(
                                0.5 * (1.0 + rng.random_range(0.0..1.0)),
                                0.5 * (1.0 + rng.random_range(0.0..1.0)),
                                0.5 * (1.0 + rng.random_range(0.0..1.0)),
                            ),
                            0.5 * rng.random_range(0.0..1.0),
                        ),
                    ));
                } else {
                    // glass
                    world.push(Sphere::new(center, 0.2, Dielectric::new(1.5)));
                }
            }
        }
    }
    world.push(Sphere::new(
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        Dielectric::new(1.5),
    ));
    world.push(Sphere::new(
        Vector3::new(-4.0, 1.0, 0.0),
        1.0,
        Metal::new(Vector3::new(0.7, 0.6, 0.5), 0.0),
    ));
    world.push(Sphere::new(
        Vector3::new(4.0, 1.0, 0.0),
        1.0,
        DiffuseLight {
            color: Vector3::new(0.5, 0.5, 0.5),
        }, // Lambertian::new(Vector3::new(0.4, 0.2, 0.1)),
    ));

    let diffuse_light = DiffuseLight {
        color: Vector3::new(0.8, 0.2, 0.2),
    };  
    world.push(Sphere::new_moving(
        Vector3::new(6.0, 1.0, 0.0),
        Vector3::new(7.0, 1.0, 0.0),
        0.5,
        // Lambertian::new(Vector3::new(0.4, 0.2, 0.5)),
        diffuse_light,
    ));
    world
}

fn main() {
    let nx = 500;
    let ny = 300;
    let ns = 100;

    let image_file = "image.ppm";
    let mut file = File::create(image_file).unwrap();
    write!(file, "P3\n{nx} {ny}\n255\n").unwrap();

    let world = random_scene();
    let look_from = Vector3::new(13.0, 2.0, 3.0);
    let look_at = Vector3::new(0.0, 0.0, 0.0);
    let focus_dist = 10.0;
    let aperture = 0.1;
    let cam = Camera::new(
        look_from,
        look_at,
        Vector3::new(0.0, 1.0, 0.0),
        20.0,
        nx as f32 / ny as f32,
        aperture,
        focus_dist,
    );

    let image = (0..ny)
        .into_par_iter()
        .rev()
        .flat_map(|y| {
            (0..nx)
                .flat_map(|x| {
                    let col: Vector3<f32> = (0..ns)
                        .map(|_| {
                            // let mut rng = SmallRng::seed_from_u64(42);
                            let mut rng = rand::rng();

                            let u = (x as f32 + rng.random_range(0.0..1.0)) / nx as f32;
                            let v = (y as f32 + rng.random_range(0.0..1.0)) / ny as f32;
                            let time = rng.random_range(0.0..1.0);
                            let ray = cam.get_ray(u, v, time);
                            cam.color(&ray, &world, 0)
                        })
                        .sum();
                    col.iter()
                        .map(|c| (255.0 * ((c / ns as f32).sqrt().clamp(0.0, 0.9999))) as u8)
                        .collect::<Vec<u8>>()
                })
                .collect::<Vec<u8>>()
        })
        .collect::<Vec<u8>>();
    for col in image.chunks(3) {
        // println!("{} {} {}", col[0], col[1], col[2]);
        file.write_all(format!("{} {} {}\n", col[0], col[1], col[2]).as_bytes())
            .unwrap();
    }
}

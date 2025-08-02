use crate::ray::Ray;
use nalgebra::Vector3;
use rand::prelude::*;
use std::f32;

fn random_in_unit_disk() -> Vector3<f32> {
    // get random using seed 42
    let mut rng = SmallRng::seed_from_u64(42);
    let unit: Vector3<f32> = Vector3::new(1.0, 1.0, 0.0);
    loop {
        let p: Vector3<f32> =
            2.0 * Vector3::new(
                rng.random_range(0.0_f32..1.0),
                rng.random_range(0.0_f32..1.0),
                0.0,
            ) - unit;
        if p.dot(&p) < 1.0 {
            return p;
        }
    }
}

pub struct Camera {
    origin: Vector3<f32>,
    lower_left_corner: Vector3<f32>,
    horizontal: Vector3<f32>,
    vertical: Vector3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    lens_radius: f32,
}

impl Camera {
    pub fn new(
        look_from: Vector3<f32>,
        look_at: Vector3<f32>,
        view_up: Vector3<f32>,
        vertical_fov: f32,
        aspect: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Self {
        let theta = vertical_fov * f32::consts::PI / 180.0;
        let half_height = focus_dist * f32::tan(theta / 2.0);
        let half_width = aspect * half_height;
        let w = (look_from - look_at).normalize();
        let u = view_up.cross(&w).normalize();
        let v = w.cross(&u);
        Camera {
            origin: look_from,
            lower_left_corner: look_from - half_width * u - half_height * v - focus_dist * w,
            horizontal: 2.0 * half_width * u,
            vertical: 2.0 * half_height * v,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    pub fn get_ray(&self, s: f32, t: f32) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        )
    }
}

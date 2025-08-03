use crate::hittable::{HitRecord, Hitable};
use crate::material::Material;
use crate::ray::Ray;
use nalgebra::Vector3;

pub struct Sphere<M: Material> {
    center: Ray,
    radius: f32,
    material: M,
}

impl<M: Material> Sphere<M> {
    pub fn new(center: Vector3<f32>, radius: f32, material: M) -> Self {
        Sphere {
            center: Ray::new(center, Vector3::zeros(), 0.0),
            radius,
            material,
        }
    }

    pub fn new_moving(center0: Vector3<f32>, center1: Vector3<f32>, radius: f32, material: M) -> Self {
        Sphere {
            center: Ray::new(center0, center1 - center0, 0.0),
            radius,
            material,
        }
    }
}

impl<M: Material> Hitable for Sphere<M> {
    fn hit<'a>(&'a self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'a>> {
        let center = self.center.point_at_parameter(ray.time());
        let oc = ray.origin() - center;
        let a = ray.direction().dot(&ray.direction());
        let b = oc.dot(&ray.direction());
        let c = oc.dot(&oc) - self.radius.powi(2);
        let discriminant = b.powi(2) - a * c;

        if discriminant > 0.0 {
            let sqrt_discriminant = discriminant.sqrt();
            let t = (-b - sqrt_discriminant) / a;
            if t < t_max && t > t_min {
                let p = ray.point_at_parameter(t);
                let normal = (p - center) / self.radius;
                return Some(HitRecord {
                    t,
                    p,
                    normal,
                    material: &self.material,
                });
            }
            let t = (-b + sqrt_discriminant) / a;
            if t < t_max && t > t_min {
                let p = ray.point_at_parameter(t);
                let normal = (p - center) / self.radius;
                return Some(HitRecord {
                    t,
                    p,
                    normal,
                    material: &self.material,
                });
            }
        }
        None
    }
}

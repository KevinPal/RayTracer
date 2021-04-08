
#include "material.h"
#include "ray.h"
#include "vector.h"
#include <math.h>


DiffuseMaterial::DiffuseMaterial(Color albedo) {
    this->albedo = albedo;
}

float DiffuseMaterial::BRDF(Ray& in, Ray& out, Vector3f& normal) {
    return 1;
}

Vector3f DiffuseMaterial::scatterRay(Vector3f& in_dir, Vector3f& normal) {
    return (normal + Vector3f::randomSphere().norm()).norm();
}

MirrorMaterial::MirrorMaterial(Color albedo) {
    this->albedo = albedo;
}

float MirrorMaterial::BRDF(Ray& in, Ray& out, Vector3f& normal) {
    return 1;
}

Vector3f MirrorMaterial::scatterRay(Vector3f& in_dir, Vector3f& normal) {
    return (normal * 2 * in_dir.dot(normal) - in_dir) * -1;
}

EmissiveMaterial::EmissiveMaterial(Color emission) {
    this->emission = emission;
}

float EmissiveMaterial::BEDF(Ray& in, Vector3f& normal) {
    return 1;
}

RefractiveMaterial::RefractiveMaterial(float IOR, float kt) {
    this->IOR = IOR;
    this->albedo = Color(1, 1, 1);
    this->kt = kt;
}

Vector3f RefractiveMaterial::scatterRay(Vector3f& in_dir, Vector3f& normal) {
    return (normal * 2 * in_dir.dot(normal) - in_dir) * -1;
}

Vector3f RefractiveMaterial::transmitRay(Vector3f& in_dir, Vector3f& normal) {

    Vector3f normal_ = normal;
    float eta = IOR;
    float cos_theta = normal.dot(in_dir * -1.0);

    if(cos_theta < 0) {
        cos_theta *= -1;
        normal_ = normal_ * -1.0;
        eta = 1.0 / eta;
    }

    float temp = 1.0 - (1.0 - cos_theta * cos_theta) / (eta * eta);
    float cos_theta2 = sqrt(temp);

    return (in_dir) / eta - normal_ * (cos_theta2 - cos_theta / eta);

}

float RefractiveMaterial::BTDF(Ray& in, Ray& out, Vector3f& normal) {

    Vector3f normal_ = normal;
    float cos_theta = normal.dot(in.direction * -1.0);
    float eta = IOR;

    if(cos_theta < 0) {
        eta = 1.0 / eta;
        normal_ = normal_ * -1.0;
        cos_theta *= -1;
    }

    float temp = 1.0 - (1.0 - cos_theta * cos_theta) / (eta * eta);
    if(temp < 0.0) {
        return 0;
    } 

    float cos_theta2 = sqrt(temp);

    Vector3f wt = (in.direction * 1.0) / eta - normal_ * (cos_theta2 - cos_theta / eta);

    return (kt / (eta * eta) / abs(normal.dot(wt)));

}
float RefractiveMaterial::BRDF(Ray& in, Ray& out, Vector3f& normal) {
    return (1 - kt);
    Vector3f scatter = scatterRay(in.direction, normal);
    return (1 - kt) / (normal.dot(scatter));
}

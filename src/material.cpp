
#include "material.h"
#include "ray.h"
#include "vector.h"

float DiffuseMaterial::BRDF(Ray& in, Ray& out, Vector3f& normal) {
    return 1;
}

Vector3f DiffuseMaterial::scatterRay(Vector3f& in_dir, Vector3f& normal) {
    return (normal + Vector3f::randomSphere().norm()).norm();
}

float MirrorMaterial::BRDF(Ray& in, Ray& out, Vector3f& normal) {
    return 1;
}

Vector3f MirrorMaterial::scatterRay(Vector3f& in_dir, Vector3f& normal) {
    return (normal * 2 * in_dir.dot(normal) - in_dir) * -1;
}

#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "vector.h"
#include "ray.h"


/* Represents a material. This class is not fully acurate yet,
 * but works for now. Specifies a color, its transperency,
 * and its reflectiveness
 *
 */
class Material {
    public:

        Color albedo;
        Color emission;

        virtual float BRDF(Ray& in, Ray& out, Vector3f& normal) = 0;
        virtual Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) = 0;

        Material() : Material(Vector3f()) {}
        Material(Color albedo_): Material(albedo_, Vector3f()) {}
        Material(Color albedo_, Color emission_): albedo(albedo_), emission(emission_) {}

};

class DiffuseMaterial : public Material {

    public:
        float BRDF(Ray& in, Ray& out, Vector3f& normal) override;
        Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) override;

        DiffuseMaterial() : Material(Vector3f()) {}
        DiffuseMaterial(Color albedo_): Material(albedo_, Vector3f()) {}
        DiffuseMaterial(Color albedo_, Color emission_): Material(albedo_, emission_) {}

};

class MirrorMaterial : public Material {

    public:
        float BRDF(Ray& in, Ray& out, Vector3f& normal) override;
        Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) override;

        MirrorMaterial() : Material(Vector3f()) {}
        MirrorMaterial(Color albedo_): Material(albedo_, Vector3f()) {}
        MirrorMaterial(Color albedo_, Color emission_): Material(albedo_, emission_) {}

};

#endif

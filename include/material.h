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
        virtual float scatterRay(Ray& in, Vector3f& normal) = 0;

        Material() : Material(Vector3f()) {}

        Material(Color albedo_): Material(albedo_, Vector3f()) {}

        Material(Color albedo_, Color emission_): albedo(albedo_), emission(emission_) {}

};

class DiffuseMaterial : public Material {

    virtual float BRDF(Ray& in, Ray& out, Vector3f& normal) override;
    virtual Ray scatterRay(Ray& in, Vector3f& normal) override;

}

#endif

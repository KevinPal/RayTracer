#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "vector.h"
#include "ray.h"

#include <stdexcept>


/* Represents a material. This class is not fully acurate yet,
 * but works for now. Specifies a color, its transperency,
 * and its reflectiveness
 *
 */
class Material {
    public:

        Color albedo;
        Color emission;

        virtual float BRDF(Ray& in, Ray& out, Vector3f& normal) { return 0; };
        virtual float BTDF(Ray& in, Ray& out, Vector3f& normal) { return 0; };
        virtual float BEDF(Ray& in, Vector3f& normal) { return 0; };

        virtual bool doesReflect() { return false; }
        virtual bool doesEmmit() { return false; }
        virtual bool doesTransmit() { return false; }

        virtual Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) {
            throw std::logic_error("Scatter Ray not implemented");
        }

        virtual Vector3f transmitRay(Vector3f& in_dir, Vector3f& normal) {
            throw std::logic_error("Scatter Ray not implemented");
        }

};

class DiffuseMaterial : public Material {

    public:
        float BRDF(Ray& in, Ray& out, Vector3f& normal) override;
        Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) override;

        bool doesReflect() { return true; }

        DiffuseMaterial(Color albedo);

};

class MirrorMaterial : public Material {

    public:
        float BRDF(Ray& in, Ray& out, Vector3f& normal) override;
        Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) override;

        bool doesReflect() { return true; }

        MirrorMaterial(Color albedo);

};

class EmissiveMaterial : public Material {

    public:
    
        float BEDF(Ray& in, Vector3f& normal);

        bool doesEmmit() { return true; }

        EmissiveMaterial(Color emission);

};

class RefractiveMaterial : public Material {

    public:

        float IOR;
        float kt;

        bool doesReflect() { return true; }
        bool doesTransmit() { return true; }

        Vector3f scatterRay(Vector3f& in_dir, Vector3f& normal) override;
        Vector3f transmitRay(Vector3f& in_dir, Vector3f& normal) override;

        float BRDF(Ray& in, Ray& out, Vector3f& normal) override;
        float BTDF(Ray& in, Ray& out, Vector3f& normal) override;

        RefractiveMaterial(float IOR, float kt);

};

#endif

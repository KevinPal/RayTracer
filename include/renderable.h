#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "material.h"
#include "ray.h"
#include "vector.h"
#include "color.h"
#include <stdio.h>

/*
 * Generic data that gets returned on a ray object intersection
 * Gets the paramateric t returned by the ray intersection,
 * the normal at which the intersection occured, and the material
 * that was hit
 */
class IntersectData {
    public:
        bool valid;
        float t;
        Vector3f normal;
        Material material;

        // Default consturctor
        IntersectData() : valid(false) {}

        // General constructor
        IntersectData(float t_, Vector3f norm_, Material m_):
            valid(true), t(t_), normal(norm_), material(m_) {}

        // Copy consturctor
        IntersectData(const IntersectData& other) :
            IntersectData(other.t, other.normal, other.material) {}
};

class AABB;

/*
 * A generic object to be rendered. Subclasses need to
 * define the intersects method, which decides if a ray
 * has it the object, and if so returns the object
 * IntersectData object containing the intersection
 * data
 */
class Renderable {
    public:

        Material material;
        AABB* bounding_box;

        // A generic renderable only needs a known material
        // At the start, any surfaces bounding surface is its self
        Renderable();
        Renderable(Material m);

        // Method to test if a ray intersects this ibject
        virtual IntersectData intersects(Ray r) = 0;
        virtual ~Renderable();

};

class AABB : public Renderable {

    public:
        Vector3f center;
        Vector3f dimensions;

        Vector3f mins;
        Vector3f maxs;
        bool valid;

        IntersectData intersects(Ray r);

        AABB();
        AABB(Vector3f center_, Vector3f dimensions_, Material m);

};


#endif

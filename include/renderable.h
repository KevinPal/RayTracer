#ifndef RENDERABLE_H
#define RENDERABLE_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "material.h"
#include "ray.h"
#include "vector.h"
#include "color.h"
#include <math.h>
#include <stdio.h>

class AABB;
class Renderable;

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
        Material* material;
        Renderable* object;

        // Default consturctor
        __host__ __device__ IntersectData() : valid(false), t(-1), object(NULL) {}

        // General constructor
        __host__ __device__ IntersectData(float t_, Vector3f norm_, Material* m_, Renderable* object_):
            valid(true), t(t_), normal(norm_), material(m_), object(object_) {}

        // Copy consturctor
        __host__ __device__ IntersectData(const IntersectData& other) :
            IntersectData(other.t, other.normal, other.material, other.object) {}

        __host__ __device__ bool did_hit() { return t >= 0; }
};


/*
 * A generic object to be rendered. Subclasses need to
 * define the intersects method, which decides if a ray
 * has it the object, and if so returns the object
 * IntersectData object containing the intersection
 * data
 */
class Renderable {
    public:

        Material* material;
        AABB* bounding_box;

        // A generic renderable only needs a known material
        // At the start, any surfaces bounding surface is its self
        Renderable();
        Renderable(Material* m);

        // Method to test if a ray intersects this ibject
        virtual IntersectData intersects(Ray r) = 0;

        virtual AABB* buildBoundingBox() = 0;
        virtual ~Renderable();

};

/*
 * An axis aligned bounding box. As a renderable, 
 * objects AABB's can also be drawn, mainly for debugging purposes
 */
class AABB : public Renderable {

    public:
        Vector3f center;
        Vector3f dimensions;

        Vector3f mins;
        Vector3f maxs;
        bool valid;

        // Checks if a ray intersects this AABB
        IntersectData intersects(Ray r);

        __device__ IntersectData intersectsGPU(Ray r);

        // Builds an AABB for this AABB.
        AABB* buildBoundingBox();

        void merge(AABB* other);

        // Invalid, empty AABB
        AABB();
        // Builds a valid AABB at the given location with the given size
        AABB(Vector3f center_, Vector3f dimensions_, Material* m);

};


#endif

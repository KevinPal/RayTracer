#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "renderable.h"
#include "mesh.h"
#include "color.h"
#include "BVH.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

/*
 * Defines a plane to be rendered
 */
class Plane : public Renderable {
    public:
        Vector3f point;
        Vector3f norm;

        // Specified a plane from a point and normal, and an optional material
        Plane(Vector3f point, Vector3f norm);
        Plane(Vector3f point, Vector3f norm, Material* material);

        // Checks if a ray intersects this plane
        IntersectData intersects(Ray r) override;
        AABB* buildBoundingBox() override;
};

/*
 * Defines a sphere to be rendered
 */
class Sphere : public Renderable {

    public:
        Vector3f center;
        float radius;
        bool invert = false;

        // Specifies a sphere from a point and raduis, and an optional material
        Sphere(Vector3f point, float radius);
        Sphere(Vector3f point, float radius, Material* material);

        __host__ __device__ IntersectData meme(Ray r);

        IntersectData intersects(Ray r) override;
        AABB* buildBoundingBox() override;
};

/*
 * Defines a triangle to be rendered
 */
class Triangle : public Renderable {

    public:
        Vector3f A;
        Vector3f B;
        Vector3f C;

        Vector3f A_normal;
        Vector3f B_normal;
        Vector3f C_normal;

        Vector2f A_tex;
        Vector2f B_tex;
        Vector2f C_tex;

        Vector3f normal;

        AABB* bounding_box;

        bool hasTexture;

        // Default constructor that just sets all 3 points to 0, 0, 0
        Triangle() {};
        // Specifies all 3 coordinates of the trinalge. Normal will be calculated
        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, Material* material);

        // Specifies all 3 coordinates of the trinalge and normals
        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, 
                Vector3f A_normal_, Vector3f B_normal_, Vector3f C_normal_,
                Material* material);

        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, 
                Vector3f A_normal_, Vector3f B_normal_, Vector3f C_normal_,
                Vector2f A_tex_, Vector2f B_tex_, Vector2f C_tex_,
                Material* material);

        // Tests of a ray intersects this trinagle
        __host__ __device__ IntersectData intersectsGPU(Ray r);
        IntersectData intersects(Ray r);
        AABB* buildBoundingBox();

};


#endif

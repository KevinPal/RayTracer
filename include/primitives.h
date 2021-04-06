#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "renderable.h"
#include "mesh.h"
#include "color.h"

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

        // Specifies a sphere from a point and raduis, and an optional material
        Sphere(Vector3f point, float radius);
        Sphere(Vector3f point, float radius, Material* material);

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

        Vector3f normal;

        // Default constructor that just sets all 3 points to 0, 0, 0
        Triangle() {};
        // Specifies all 3 coordinates of the trinalge. Normal will be calculated
        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, Material* material);

        // Specifies all 3 coordinates of the trinalge and normals
        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, 
                Vector3f A_normal_, Vector3f B_normal_, Vector3f C_normal_,
                Material* material);

        // Tests of a ray intersects this trinagle
        IntersectData intersects(Ray r);
        AABB* buildBoundingBox() override;

};

/*
 * Definnes an arbitrary 3d prism to be rendered. Gets broken down
 * into 12 triangles, which is placed into a mesh to be rendered
 */
class Prism : public Mesh {

    public:
        Vector3f center;
        Vector3f up;
        Vector3f right;
        Vector3f dimensions;

        // Defines a prism based on how to go "up" and "right". The 3rd vector is perpendicular
        // to both of these
        Prism(Vector3f center_, Vector3f up_, Vector3f right_, Vector3f dimensions_, Material* material);
        ~Prism();

        Triangle* triangles[12];

};

#endif

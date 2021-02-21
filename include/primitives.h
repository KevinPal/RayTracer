#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "renderable.h"
#include "mesh.h"
#include "color.h"

class Plane : public Renderable {
    public:
        Vector3f point;
        Vector3f norm;

        Plane(Vector3f point, Vector3f norm);
        Plane(Vector3f point, Vector3f norm, Material material);

        IntersectData intersects(Ray r) override;
};

class Sphere : public Renderable {

    public:
        Vector3f center;
        float radius;

        Sphere(Vector3f point, float radius);
        Sphere(Vector3f point, float radius, Material material);

        IntersectData intersects(Ray r) override;
};

class Triangle : public Renderable {

    public:
        Vector3f normal;

        Vector3f A;
        Vector3f B;
        Vector3f C;

        Triangle() {};
        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, Material material);
        IntersectData intersects(Ray r);

};

class Prism : public Mesh {

    public:
        Vector3f center;
        Vector3f up;
        Vector3f right;
        Vector3f dimensions;

        Prism(Vector3f center_, Vector3f up_, Vector3f right_, Vector3f dimensions_, Material material);

        Triangle triangles[12];

};

#endif

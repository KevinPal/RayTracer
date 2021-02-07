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
        Plane(Vector3f point, Vector3f norm, Color color);

        IntersectData intersects(Ray r) override;
};

class Sphere : public Renderable {

    public:
        Vector3f center;
        float radius;

        Sphere(Vector3f point, float radius);
        Sphere(Vector3f point, float radius, Color color);

        IntersectData intersects(Ray r) override;
};

class Triangle : public Renderable {

    public:
        Vector3f normal;

        Vector3f A;
        Vector3f B;
        Vector3f C;

        Triangle() {};
        Triangle(Vector3f A_, Vector3f B_, Vector3f C_, Color color_);
        IntersectData intersects(Ray r);

};

class Prism : public Mesh {

    public:
        Vector3f center;
        Vector3f up;
        Vector3f right;
        Vector3f dimensions;

        Prism(Vector3f center_, Vector3f up_, Vector3f right_, Vector3f dimensions_, Color color_);

        Triangle triangles[12];

};

#endif

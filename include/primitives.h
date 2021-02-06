#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "renderable.h"
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

#endif

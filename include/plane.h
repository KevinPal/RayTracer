#ifndef PLANE_H
#define PLANE_H

#include "renderable.h"
#include "color.h"

class Plane : Renderable {
    public:
        Vector3f point;
        Vector3f norm;
        Color color;

        Plane(Vector3f point, Vector3f norm);
        Plane(Vector3f point, Vector3f norm, Color color);

        IntersectData intersects(Ray r) override;
};

#endif

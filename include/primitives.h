#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "renderable.h"
#include "color.h"

class Plane : public Renderable {
    public:
        Vector3f point;
        Vector3f norm;
        Color color;

        Plane(Vector3f point, Vector3f norm);
        Plane(Vector3f point, Vector3f norm, Color color);

        IntersectData intersects(Ray r) override;
};

#endif

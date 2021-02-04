#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "ray.h"
#include "vector.h"
#include "color.h"

class IntersectData {
    public:
        float t;
        Color color;
        Vector3f normal;

        IntersectData() {}
};

class Renderable {
    public:
        virtual IntersectData intersects(Ray r) = 0;
};


#endif

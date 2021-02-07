#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "ray.h"
#include "vector.h"
#include "color.h"

class IntersectData {
    public:
        float t;
        Vector3f normal;
        Color color;

        IntersectData() {}
        IntersectData(float t_, Vector3f norm_, Color c_):
            t(t_), normal(norm_), color(c_) {}
        IntersectData(const IntersectData& other) :
            IntersectData(other.t, other.normal, other.color) {}
};

class Renderable {
    public:

        Color color;

        Renderable() {}
        Renderable(Color c) : color(c) {}
        virtual IntersectData intersects(Ray r) = 0;
};


#endif

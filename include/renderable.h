#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "material.h"
#include "ray.h"
#include "vector.h"
#include "color.h"
#include <stdio.h>

class IntersectData {
    public:
        float t;
        Vector3f normal;
        Material material;

        IntersectData() {}
        IntersectData(float t_, Vector3f norm_, Material m_):
            t(t_), normal(norm_), material(m_) {}
        IntersectData(const IntersectData& other) :
            IntersectData(other.t, other.normal, other.material) {}
};

class Renderable {
    public:

        Material material;

        Renderable() {}
        Renderable(Material m) : material(m) { }
        virtual IntersectData intersects(Ray r) = 0;
};


#endif

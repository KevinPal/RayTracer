#ifndef MESH_H
#define MESH_H

#include "renderable.h"

#include <vector>

class Mesh : public Renderable {

    public:
        std::vector<Renderable*> objects;
        Mesh() : Renderable(Color()) {}
        IntersectData intersects(Ray r);

        void addObject(Renderable* obj);
};

#endif

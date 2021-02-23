#ifndef MESH_H
#define MESH_H

#include "renderable.h"

#include <vector>

/*
 * The mesh class represents a group of other objects that
 * can be rendered. An intersection check tests all the object
 * within this mesh. Note that a mesh its self can contain
 * other meshs, and the entire topmost scene should be
 * a singular mesh
 */
class Mesh : public Renderable {

    public:
        std::vector<Renderable*> objects;

        // Creates a empty mesh
        Mesh() : Renderable(Color()) {}

        // Tests if a ray intersects any of the objects
        // in this mesh
        IntersectData intersects(Ray r);

        // Adds a renderable to this mesh
        void addObject(Renderable* obj);
};

#endif

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

    protected:

        bool built_bounding_box;

    public:
        // Objects is an array of things that have bounding boxes
        // large objects are things that don't, such as planes
        std::vector<Renderable*> objects;
        std::vector<Renderable*> large_objects;

        // Creates a empty mesh
        Mesh();

        // Creates a mesh from a given iterator
        Mesh(std::vector<Renderable*>::iterator start, std::vector<Renderable*>::iterator end);

        // Tests if a ray intersects any of the objects
        // in this mesh
        IntersectData intersects(Ray r);

        IntersectData intersects_large(Ray r);

        // Adds a renderable to this mesh
        void addObject(Renderable* obj);

        void buildBoundingBox(void);
};


#endif

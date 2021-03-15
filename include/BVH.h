#ifndef BVH_H
#define BVH_H

#include "mesh.h"
#include "renderable.h"

class BVHNode : public Mesh {

    public:
        BVHNode* left;
        BVHNode* right;
        int leaf_size;

        BVHNode(int leaf_size_) : left(NULL), right(NULL), leaf_size(leaf_size_) {}

        BVHNode(int leaf_size_, std::vector<Renderable*>::iterator start, std::vector<Renderable*>::iterator end)
            : Mesh(start, end), left(NULL), right(NULL), leaf_size(leaf_size_) {}

        IntersectData intersects(Ray r);

        void partition();
};

#endif

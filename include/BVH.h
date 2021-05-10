#ifndef BVH_H
#define BVH_H

#include "mesh.h"
#include "renderable.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

// Class to represent the BVHNode. As a subclass of mesh, it 
// still contains an array of renderables at any point, so
// any node could be drawn using underlying mesh functions.
class BVHNode : public Mesh {

    public:
        // Left and right pointers, as well as leaf node size for when
        // to stop paritioning
        BVHNode* left;
        BVHNode* right;
        int leaf_size;

        // Mesh constructors with added leaf size paramater. Defaults left and right to null
        BVHNode(Material* material_, int leaf_size_) :
            Mesh(material_), left(NULL), right(NULL), leaf_size(leaf_size_) {}
        BVHNode(Material* material_, int leaf_size_, std::vector<Renderable*>::iterator start, std::vector<Renderable*>::iterator end)
            : Mesh(material_, start, end), left(NULL), right(NULL), leaf_size(leaf_size_) {}

        // Check if a ray intersects this BVH
        __host__ __device__ IntersectData intersects(Ray r);

        // Recursively partition this BVH
        void partition();
};

#endif

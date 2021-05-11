#ifndef BVH_H
#define BVH_H

#include "mesh.h"
#include "renderable.h"
#include "primitives.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

class BVHNodeGPU;
class Triangle;

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
        IntersectData intersects(Ray r);

        // Recursively partition this BVH
        void partition();

        BVHNodeGPU* deepCudaCopy(std::vector<void*>);
};


class BVHNodeGPU {

    public:
        BVHNodeGPU* left;
        BVHNodeGPU* right;

        Triangle* triangles;
        int num_triangles;

        AABB bounding_box;

        BVHNodeGPU() {}

        BVHNodeGPU(BVHNodeGPU* left_, BVHNodeGPU* right_, Triangle* triangles_, int num_triangles_, AABB bounding_box_):
            left(left_), right(right_), triangles(triangles_), num_triangles(num_triangles_), bounding_box(bounding_box_) {}

        __device__ IntersectData intersects(Ray r);
        __device__ IntersectData intersectsBrute(Ray r);



};

#endif

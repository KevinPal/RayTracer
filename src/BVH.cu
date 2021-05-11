

#include "mesh.h"
#include "renderable.h"
#include "BVH.h"
#include <algorithm>
#include <stdio.h>
#include <cassert>

#define DO_BVH 1

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("%s%s\n", "CUDA error: ", cudaGetErrorString(err));              \
      printf("%s%s\n", "Failed to run stmt ", #stmt);                         \
    }                                                                     \
  } while (0)

/*
 * Checks if a ray intersects this BVH Node. Will
 * recusively check left and right children if the 
 * ray intersects their AABBs, otherwise will
 * defer to the underlying mesh's intersection methods
 */
IntersectData BVHNode::intersects(Ray r) {

    if(!DO_BVH) {
        // Use brute force if we don't want BVH
        return Mesh::intersects(r);
    } else {

        // Both should be null or not null
        assert(!((this->left == NULL) ^ (this->right == NULL)));

        // Base case, check the elements in the mesh, as well as
        // the global "slow" primitives (i.e planes). This is 
        // already handeled in the mesh brute force case
        if(this->left == NULL && this->right == NULL) {
            return Mesh::intersects(r);
        } else {
            // Check if we hit left and/or right boxes
            IntersectData left_bb = this->left->bounding_box->intersects(r);
            IntersectData right_bb = this->right->bounding_box->intersects(r);

            if(!right_bb.did_hit()) {
                if(!left_bb.did_hit()) {
                    // Neither hit
                    return Mesh::intersects_large(r);
                } else {
                    // Left hit
                    return this->left->intersects(r);
                }
            } else {
                // Left didnt hit but right did
                if(!left_bb.did_hit()) {
                    return this->right->intersects(r);
                } else {
                    // Both hit, take closest
                    IntersectData left_inter = this->left->intersects(r);
                    IntersectData right_inter = this->right->intersects(r);

                    if(left_inter.did_hit() && !right_inter.did_hit()) {
                        return left_inter;
                    } else if(!left_inter.did_hit() && right_inter.did_hit()) {
                        return right_inter;
                    } else if(left_inter.did_hit() && right_inter.did_hit()) {
                        if(left_inter.t <= right_inter.t) {
                            return left_inter;
                        } else {
                            return right_inter;
                        }
                    } else {
                        return Mesh::intersects_large(r);
                    }
                }
            }
        }
    }
}

BVHNodeGPU* BVHNode::deepCudaCopy(std::vector<void*> cudaPtrs) {

    assert(~(left == NULL) ^ (right == NULL));

    BVHNodeGPU cpu_node;

    if(left != NULL && right != NULL) {
        BVHNodeGPU* gpu_left = left->deepCudaCopy(cudaPtrs);
        BVHNodeGPU* gpu_right = right->deepCudaCopy(cudaPtrs);

        cpu_node = BVHNodeGPU(gpu_left, gpu_right, NULL, 0, *(this->bounding_box));
    } else {

        Triangle* deviceTriangle;

        wbCheck(cudaMalloc((void**) &deviceTriangle, sizeof(Triangle) * this->objects.size()));
        cudaPtrs.push_back(deviceTriangle);

        for(int i = 0; i < objects.size(); i++) {
            wbCheck(cudaMemcpy((void*) &(deviceTriangle[i]), objects[i], sizeof(Triangle), cudaMemcpyHostToDevice));
        }

        cpu_node = BVHNodeGPU(NULL, NULL, deviceTriangle, objects.size(), *(this->bounding_box));
    }

    BVHNodeGPU* gpu_node;

    wbCheck(cudaMalloc((void**) &gpu_node, sizeof(BVHNodeGPU)));
    wbCheck(cudaMemcpy(gpu_node, &cpu_node, sizeof(BVHNodeGPU), cudaMemcpyHostToDevice));

    cudaPtrs.push_back(gpu_node);

    return gpu_node;
}

__device__ IntersectData BVHNodeGPU::intersectsBrute(Ray ray) {

    bool hit = false;
    IntersectData min_hit;
    min_hit.t =  -1;
    for(int i = 0; i < num_triangles; i++) {

        IntersectData data = triangles[i].meme(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit.t = data.t;
            min_hit.normal = data.normal;
        }
    }

    return min_hit;

}

__device__ IntersectData BVHNodeGPU::intersects(Ray ray) {

    // Both should be null or not null
    assert(!((this->left == NULL) ^ (this->right == NULL)));

    bool hit = false;
    IntersectData min_hit;
    min_hit.t = -1;


    if(!DO_BVH) {
        // Use brute force if we don't want BVH
        return intersectsBrute(ray);
    } else {

        BVHNodeGPU* stack[32];
        for(int i = 0; i < 32; i++) {
            stack[i] = NULL;
        }
        int stackPtr = 0;

        stack[stackPtr++] = this;

        while(stackPtr > 0) {

            assert(stackPtr < 30);

            // Pop
            BVHNodeGPU* curr = stack[stackPtr - 1];
            stackPtr--;

            if(curr->left == NULL && curr->right == NULL) {
                // Leaf node
                IntersectData data = curr->intersectsBrute(ray);
                if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
                    hit = true;
                    min_hit.t = data.t;
                    min_hit.normal = data.normal;
                }
            } else {
                IntersectData left_bb = curr->left->bounding_box.intersectsGPU(ray);
                IntersectData right_bb = curr->right->bounding_box.intersectsGPU(ray);

                if(!right_bb.did_hit()) {
                    if(!left_bb.did_hit()) {
                        // Neither hit
                        continue;
                    } else {
                        // Left hit
                        //return this->left->intersects(r);
                        // Push left

                        stack[stackPtr++] = curr->left;
                        continue;
                    }
                } else {
                    // Left didnt hit but right did
                    if(!left_bb.did_hit()) {
                        //return this->right->intersects(r);
                        // Push right
                        stack[stackPtr++] = curr->right;
                        continue;
                    } else {
                        // Both hit, push both
                        //IntersectData left_inter = this->left->intersects(ray);
                        //IntersectData right_inter = this->right->intersects(ray);

                        stack[stackPtr++] = curr->right;
                        stack[stackPtr++] = curr->left;
                        continue;


                        /*
                        if(left_inter.did_hit() && !right_inter.did_hit()) {
                            return left_inter;
                        } else if(!left_inter.did_hit() && right_inter.did_hit()) {
                            return right_inter;
                        } else if(left_inter.did_hit() && right_inter.did_hit()) {
                            if(left_inter.t <= right_inter.t) {
                                return left_inter;
                            } else {
                                return right_inter;
                            }
                        } else {
                            return no_hit;
                        }
                        */
                    }
                }
            }
        }

        return min_hit;
    }
}

/*
 * Partitions this BVHS. Attemps to splits the underlying mesh
 * into two meshes along the axis with the greatest spread,
 * then partitions based on the median. Stops partitioning
 * if the underlying mesh has less than left_size elements, 
 * or if partitioning is meaningless (i.e all elements in the
 * same spot). Recursively handles the new BVHNodes
 */
void BVHNode::partition() {

    // If the underlying mesh doesnt have an AABB yet, build it
    if(!this->built_bounding_box) {
        this->buildBoundingBox();
    }

    // Base cases
    if(this->objects.size() < this->leaf_size) {
        return;
    }

    // Find axis with greatest spread
    int axis = 1;
    float last_size = -1;
    for(int dim = 0; dim < 3; dim++) {
        float size = this->bounding_box->maxs[dim] - this->bounding_box->mins[dim] ;
        if(size > last_size) {
            last_size = size;
            axis = dim;
        }
    }
    
    printf("Splitting on %d with size %f\n", axis, last_size);

    // Use nthstort to partition based on median
    std::nth_element(objects.begin(), objects.begin() + objects.size()/2, objects.end(),
            [&axis](const Renderable* A, const Renderable* B) {
                // Define the comparison function to select along the axis from above
                return A->bounding_box->center[axis] < B->bounding_box->center[axis];
            }
    );

    float median = objects[objects.size()/2]->bounding_box->center[axis];

    // Create left and right nodes
    //this->left = new BVHNode(leaf_size, objects.begin(), objects.begin() + objects.size()/2);
    //this->right = new BVHNode(leaf_size, objects.begin() + objects.size()/2, objects.end());
    this->left = new BVHNode(this->material, leaf_size);
    this->right = new BVHNode(this->material, leaf_size);


    for(Renderable* r : objects) {
        if(r->bounding_box->center[axis] <= median) {
            this->left->addObject(r);
        } else {
            this->right->addObject(r);
        }
    }

    // Unable to properly partiton. Just make this a leaf node
    if(static_cast<BVHNode*>(this->left)->objects.size() == 0 || static_cast<BVHNode*>(this->right)->objects.size() == 0) {
        delete this->left;
        delete this->right;

        this->left = NULL;
        this->right = NULL;
        printf("Forcing leaf\n");
    } else {
        printf("Left size: %d right size: %d\n", static_cast<BVHNode*>(this->left)->objects.size(), static_cast<BVHNode*>(this->right)->objects.size());

        ((BVHNode*) (this->left ))->large_objects.assign(this->large_objects.begin(), this->large_objects.end());
        ((BVHNode*) (this->right))->large_objects.assign(this->large_objects.begin(), this->large_objects.end());

        // Recurse
        ((BVHNode*) (this->left ))->partition();
        ((BVHNode*) (this->right))->partition();
    }

}

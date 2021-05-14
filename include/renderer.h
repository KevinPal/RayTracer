#ifndef RENDERER_H
#define RENDERER_H

#include <vector>

#include "ray.h"
#include "primitives.h"
#include "renderable.h"
#include "renderer.h"
#include "mesh.h"
#include "color.h"
#include "BVH.h"
#include "matrix.h"

typedef struct device_data_t {
    std::vector<void*> ptrs;
    Ray* deviceRayBuffer;
    Ray* deviceShadowRayBuffer;
    IntersectData* deviceIntersectBuffer;
    IntersectData* deviceShadowIntersectBuffer;
    unsigned char* deviceOutput;
    BVHNodeGPU* scene;

    Matrix44* deviceTransform;
    Matrix44* deviceInvTransform;

    char* deviceTexture;
} device_data_t;

// Renders a ray passing through the scene. Depth is
// the number of ray bounces we want to do
Color renderRay(Ray r, Mesh* scene, Mesh* lighting, int depth, long* rays);

__global__ void renderRaysKernel(Ray* rays, unsigned char* output, IntersectData* primaryIntersect, IntersectData* shadowIntersect, int width, int height, BVHNodeGPU* scene, Matrix44* transform, Matrix44* invTransform, char* textureMem, int textureWidth, int textureHeight);
__global__ void intersectKernel(Ray* rays, IntersectData* output, Ray* shadow_rays, int width, int height, BVHNodeGPU* scene, Matrix44* transform, Matrix44* invTransform);

__host__ void renderRaysInit(Ray* rays, BVHNode* scene, device_data_t* device_data, int width, int height, char* hostTexture, int textureWidth, int textureHeight);
__host__ void renderRays(device_data_t* device_data, unsigned char* output, int width, int height, int textureWidth, int textureHeight);
__host__ void renderRaysCleanup(device_data_t* device_data);

#endif

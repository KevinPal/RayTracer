#ifndef RENDERER_H
#define RENDERER_H

#include "ray.h"
#include "primitives.h"
#include "renderable.h"
#include "renderer.h"
#include "mesh.h"
#include "color.h"

// Renders a ray passing through the scene. Depth is
// the number of ray bounces we want to do
Color renderRay(Ray r, Mesh* scene, Mesh* lighting, int depth, long* rays);
__host__ void renderRays(Ray* rays, unsigned char* output, int width, int height, Sphere* s);
__global__ void renderRaysKernel(Ray* rays, unsigned char* output, int width, int height, Sphere* s);

#endif

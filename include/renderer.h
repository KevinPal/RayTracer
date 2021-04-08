#ifndef RENDERER_H
#define RENDERER_H

#include "ray.h"
#include "renderable.h"
#include "renderer.h"
#include "mesh.h"
#include "color.h"

// Renders a ray passing through the scene. Depth is
// the number of ray bounces we want to do
Color renderRay(Ray r, Mesh* scene, Mesh* lighting, int depth, long* rays);

#endif

#ifndef RENDERER_H
#define RENDERER_H

#include "ray.h"
#include "renderable.h"
#include "renderer.h"
#include "mesh.h"

IntersectData renderRay(Ray r, Mesh* scene, int depth);

#endif


#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "renderable.h"

// Default renderable constructor, does not have a bounding box
Renderable::Renderable() : bounding_box(NULL) {}

// Renderable with known material, does not have a bounding box
Renderable::Renderable(Material* m) :
   bounding_box(NULL),  material(m) { }

// On deconstruction of renderable, also delete the bounding box 
// if it exsits
Renderable::~Renderable() {
    if(this->bounding_box && (this->bounding_box != this)) {
        printf("Deleting aabb at %p\n", this->bounding_box);
        delete this->bounding_box;
    }
}

// Default AABB, set as invalid
AABB::AABB() :
    valid(false) {
    this->bounding_box = NULL;//buildBoundingBox();
}

// Generic AABB located at a point with given dimensions
AABB::AABB(Vector3f center_, Vector3f dimensions_, Material* m) :
    Renderable(m), valid(true), center(center_), dimensions(dimensions_) {
   
    this->bounding_box = this;
    this->mins = center - dimensions_ / 2.0;
    this->maxs = center + dimensions_ / 2.0;
}

// The AABB of an AABB is its self
AABB* AABB::buildBoundingBox() {
    printf("self aabb: %p\n", this);
    return this;
}

/* 
 * Based off of "Fast Ray-Box Intersection" by Andrew Woo
 * from "Graphics Gems", Academic Press, 1990.
 * Source: https://web.archive.org/web/20090803054252/http://tog.acm.org/resources/GraphicsGems/gems/RayBox.c
 * 
 * Only 3 sides of a plane can ever be visible from any angle. We find those
 * 3 planes, and do comparisons against those. Checks common cases first, such
 * as the ray being inside the box, or the ray completely pointing the wrong way,
 * before handling the more intensive cases
*/
IntersectData AABB::intersects(Ray ray) {
    bool inside = true;
    char quadrant[3];
    float T[3];
    float candidate_plane[3];
    int max_plane;

    const char right = 0;
    const char left = 1;
    const char middle = 2;

    IntersectData out;
    out.material = this->material;
    out.t = -1;
    out.object = this;

    // Find where the ray is on the left, right, or in the middle
    // of each dimension. Based on that, we reduce this to 3 plane 
    // intersection checks
    for(int i = 0; i < 3; i++) {
        if(ray.origin[i] < this->mins[i]) {
            quadrant[i] = left;
            candidate_plane[i] = this->mins[i];
            inside = false;
        } else if(ray.origin[i] > this->maxs[i]) {
            quadrant[i] = right;
            candidate_plane[i] = this->maxs[i];
            inside = false;
        } else {
            quadrant[i] = middle;
        }
    }

    if(inside) {
        out.t = 0;
        return out;
    }

    // Get minimum distance to the 3 planes along each axis
    for(int i = 0; i < 3; i++) {
        if(quadrant[i] != middle && ray.direction[i] != 0) {
            T[i] = (candidate_plane[i] - ray.origin[i]) / ray.direction[i];
        } else {
            // If the ray is paralell to this axis, it will never hit.
            // If it is inbetween the max and min, it will hit one
            // of the other sides of the box first
            T[i] = -1;
        }
    }

    // Get the farthest plane that hit
    max_plane = -1;
    for(int i = 0; i < 3; i++) {
        if((T[max_plane] < T[i] || max_plane == -1) && T[i] > 0) {
            max_plane = i;
        }
    }

    // Didn't hit any of the 3 planes
    if(max_plane == -1) {
        return out;
    }

    // Make sure the other sides hit
    for(int i = 0; i < 3; i++) {
        if(max_plane == i) {
            continue;
        } else {
            float coord = ray.origin[i] + T[max_plane] * ray.direction[i];
            // Didnt hit this side
            if(coord < this->mins[i] || coord > this->maxs[i]) {
                return out;
            }
        }
    }
    
    out.t = T[max_plane];
    return out;
}

// Copy of intersects
__device__ IntersectData AABB::intersectsGPU(Ray ray) {
    bool inside = true;
    char quadrant[3];
    float T[3];
    float candidate_plane[3];
    int max_plane;

    const char right = 0;
    const char left = 1;
    const char middle = 2;

    IntersectData out;
    out.t = -1;

    // Find where the ray is on the left, right, or in the middle
    // of each dimension. Based on that, we reduce this to 3 plane 
    // intersection checks
    for(int i = 0; i < 3; i++) {
        if(ray.origin[i] < this->mins[i]) {
            quadrant[i] = left;
            candidate_plane[i] = this->mins[i];
            inside = false;
        } else if(ray.origin[i] > this->maxs[i]) {
            quadrant[i] = right;
            candidate_plane[i] = this->maxs[i];
            inside = false;
        } else {
            quadrant[i] = middle;
        }
    }

    if(inside) {
        out.t = 0;
        return out;
    }

    // Get minimum distance to the 3 planes along each axis
    for(int i = 0; i < 3; i++) {
        if(quadrant[i] != middle && ray.direction[i] != 0) {
            T[i] = (candidate_plane[i] - ray.origin[i]) / ray.direction[i];
        } else {
            // If the ray is paralell to this axis, it will never hit.
            // If it is inbetween the max and min, it will hit one
            // of the other sides of the box first
            T[i] = -1;
        }
    }

    // Get the farthest plane that hit
    max_plane = -1;
    for(int i = 0; i < 3; i++) {
        if((T[max_plane] < T[i] || max_plane == -1) && T[i] > 0) {
            max_plane = i;
        }
    }

    // Didn't hit any of the 3 planes
    if(max_plane == -1) {
        return out;
    }

    // Make sure the other sides hit
    for(int i = 0; i < 3; i++) {
        if(max_plane == i) {
            continue;
        } else {
            float coord = ray.origin[i] + T[max_plane] * ray.direction[i];
            // Didnt hit this side
            if(coord < this->mins[i] || coord > this->maxs[i]) {
                return out;
            }
        }
    }
    
    out.t = T[max_plane];
    return out;
}

/*
 * Merges another AABB into this one. Based on the valididty of either
 * AABB, this AABB could be unchanged, equivelent to the other AABB,
 * or a new AABB that can contain both AABBs.
 */
void AABB::merge(AABB* other) {
    //printf("merging this: %d other: %d \n", valid, other.valid);
    if(!other->valid && valid) {
        return;
    } else if(other->valid && !valid) {
        this->maxs = other->maxs;
        this->mins = other->mins;
        this->dimensions = other->dimensions;
        this->center = other->center;
        this->valid = true;
    } else if(valid && other->valid) {

        for(int i = 0; i < 3; i++) {
            this->maxs[i] = std::max(this->maxs[i], other->maxs[i]);
            this->mins[i] = std::min(this->mins[i], other->mins[i]);
        }

        this->center = (this->maxs + this->mins) / 2.0;
        this->dimensions = (this->maxs - this->mins);
    }
}


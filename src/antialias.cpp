#include "antialias.h"
#include "camera.h"
#include "ray.h"
#include <cassert>

#include <stdio.h>

GridAntiAliaser::GridAntiAliaser(const RayIterator* iter, int grid_size_):
        AntiAliaser(iter), grid_size(grid_size_) {

    assert(iter != NULL);

    //this->primary_ray = **iter;
}

Ray GridAntiAliaser::operator*() const {
    return ray_iter->camera.project(world_cord);
}

GridAntiAliaser& GridAntiAliaser::operator++() {

    if(screen_offset.x >= 1) {
        screen_offset = Vector2f(0, screen_offset.y + 1.0 / grid_size);
    } else {
        screen_offset.x += 1.0 / grid_size;
    }

    world_cord = ray_iter->getWorldCord() + (ray_iter->camera.right * screen_offset.x + ray_iter->camera.up * screen_offset.y);
    return *this;
}

bool GridAntiAliaser::isDone() {
    return screen_offset.x >= 1 && screen_offset.y >= 1;
}


#include "antialias.h"
#include "camera.h"
#include "ray.h"
#include <cassert>

#include <stdio.h>

// Grid antialias constructor
GridAntiAliaser::GridAntiAliaser(const RayIterator* iter, int grid_size_):
        AntiAliaser(iter), grid_size(grid_size_) {
    assert(iter != NULL);
}

// Projects the current world coord through the camera
Ray GridAntiAliaser::operator*() const {
    Vector2f screen_cord = ray_iter->getScreenCord();
    return ray_iter->camera.project(world_cord);
}

// Creates the next coord in a grid like fasion
GridAntiAliaser& GridAntiAliaser::operator++() {

    // If we hit the right hand side of the pixel, loop left and move one down
    if(screen_offset.x >= (1 - 1.0/grid_size)) {
        screen_offset = Vector2f(0, screen_offset.y + 1.0 / grid_size);
    } else {
        screen_offset.x += 1.0 / grid_size;
    }

    // Translate the screen offset into a world offset based on the cameras basis vectors
    Vector2f screen_cord = ray_iter->getScreenCord() + screen_offset;


    Vector2f global_screen_offset = (screen_cord - (ray_iter->camera.resolution / 2));
    world_cord = ray_iter->camera.location + (ray_iter->camera.right * global_screen_offset.x - ray_iter->camera.up * global_screen_offset.y);
    return *this;
}

// Checks if we are done rendering every subpixel
bool GridAntiAliaser::isDone() {
    return screen_offset.x >= (1-1.0/grid_size) && screen_offset.y >= 1 - 1.0/grid_size;
}

// Grid antialias constructor
RandomAntiAliaser::RandomAntiAliaser(const RayIterator* iter, int grid_size_, int num_samples_):
        AntiAliaser(iter), grid_size(grid_size_), num_samples(num_samples_) {
    assert(iter != NULL);
}

// Projects the current world coord through the camera
Ray RandomAntiAliaser::operator*() const {
    Vector2f screen_cord = ray_iter->getScreenCord();
    return ray_iter->camera.project(world_cord);
}

// Creates the next coord in a grid like fasion
RandomAntiAliaser& RandomAntiAliaser::operator++() {

    // If we hit the right hand side of the pixel, loop left and move one down
    if(screen_offset.x >= (1 - 1.0/grid_size)) {
        screen_offset = Vector2f(0, screen_offset.y + 1.0 / grid_size);
    } else {
        screen_offset.x += 1.0 / grid_size;
    }

    // Translate the screen offset into a world offset based on the cameras basis vectors
    Vector2f screen_cord = ray_iter->getScreenCord() + screen_offset;


    Vector2f global_screen_offset = (screen_cord - (ray_iter->camera.resolution / 2));
    world_cord = ray_iter->camera.location + (ray_iter->camera.right * global_screen_offset.x - ray_iter->camera.up * global_screen_offset.y);
    return *this;
}

// Checks if we are done rendering every subpixel
bool RandomAntiAliaser::isDone() {
    return num_samples <= 0;
}


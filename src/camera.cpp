
#include "camera.h"
#include "vector.h"

// Generic camera constructor, calculates this cameras basis
// in world coordinates 
Camera::Camera(Vector3f location_, Vector3f up_, Vector3f right_,
       Vector2f size_, Vector2f resolution_)
        : location(location_), up(up_), right(right_),
          size(size_), resolution(resolution_) {

    Vector2f world_stride = this->size / this->resolution;
    // Invert stride's y since +y is up in world but -y is up in screen
    this->up = this->up.normalize() * world_stride.y;
    this->right = this->right.normalize() * world_stride.x;
    this->norm = right.cross(up).normalize();

};

// Ray iterator constructor, gets starting offset and world coordinates
RayIterator::RayIterator(Camera& camera_, int x, int y) : camera(camera_) {

    screen_cord = Vector2f(x, y);

    Vector2f screen_offset = (screen_cord - (camera.resolution / 2));
    // Minus y here since screen coords up is -y
    world_cord = camera.location + (camera.right * screen_offset.x - camera.up * screen_offset.y);
}

// Projects the current world coordinate through the camera
Ray RayIterator::operator*() const {
    return camera.project(world_cord);
}

// Really shouldnt use this, but post increment operator
RayIterator RayIterator::operator++(int) {
    RayIterator tmp(*this);
    this->operator++();
    return tmp;
}

// Moves the ray iterator to the next pixel, and calculates the new
// screen and world offsets
RayIterator& RayIterator::operator++() {

    // If we hit the right hand side of the screen, we wrap to the left
    // and move one down, otherwise move one right
    if(screen_cord.x == camera.resolution.x) {
        screen_cord = Vector2f(0, screen_cord.y + 1);
        // Move all the way back left, then down 1
        world_cord = world_cord - (camera.right * camera.resolution.x) - camera.up;
    } else {
        screen_cord.x += 1;
        world_cord = world_cord + camera.right;
    }

    Vector2f screen_offset = (screen_cord - (camera.resolution / 2));
    // Minus y here since screen coords up is -y
    world_cord = camera.location + (camera.right * screen_offset.x - camera.up * screen_offset.y);

    return *this;
};

// If two iterators are rendering the same screen pixel, we will consider them equal. This is
// not true if there are multiple cameras but we will leave that out for now
bool RayIterator::operator==(const RayIterator& other) {
    // Need to check for different cameras but will ignore for now
    return this->screen_cord == other.screen_cord;
}
// See above
bool RayIterator::operator!=(const RayIterator& other) {
    return !(*this == other);
}

// Ortho constructor camera
OrthoCamera::OrthoCamera(Vector3f location_, Vector3f up_, Vector3f right_,
        Vector2f size_, Vector2f resolution_) :
    Camera(location_, up_, right_, size_, resolution_) {}

// To project an orthographic ray through a point on the view plane specified
// by its world coordinate, we just send a ray out normal to the view plane
Ray OrthoCamera::project(Vector3f world_cord) {
    return Ray(world_cord + Vector3f(0.5, 0.5, 0), norm);
}

// Perspective camera constructor
PerspectiveCamera::PerspectiveCamera(Vector3f location_, Vector3f up_, Vector3f right_,
        Vector2f size_, Vector2f resolution_, Vector3f eye_) :
    Camera(location_, up_, right_, size_, resolution_), eye(eye_) {}

// To do a perspective projection, we send out a ray from the camera eye
// to the given world coord. 
Ray PerspectiveCamera::project(Vector3f world_cord) {
    Ray r;
    r.fromPoints(eye, world_cord);
    //r.origin = world_cord;
    r.direction.normalize();
    return r;
}

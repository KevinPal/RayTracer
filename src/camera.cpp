
#include "camera.h"
#include "vector.h"

Camera::Camera(Vector3f location_, Vector3f up_, Vector3f right_,
       Vector2f size_, Vector2f resolution_)
        : location(location_), up(up_), right(right_),
          size(size_), resolution(resolution_) {

    Vector2f world_stride = this->size / this->resolution;
    // Invert stride's y since +y is up in world but -y is up in screen
    //
    //
    //up.normalize().print();
    //right.print();
    this->up = this->up.normalize() * world_stride.y;
    this->right = this->right.normalize() * world_stride.x;
    //this->norm = up.cross(right).normalize();
    this->norm = right.cross(up).normalize();

};

RayIterator::RayIterator(Camera& camera_, int x, int y) : camera(camera_) {

    screen_cord = Vector2f(x, y);

    Vector2f screen_offset = (screen_cord - (camera.resolution / 2));
    //camera.resolution.print();
    // Minus y here since screen coords up is -y
    world_cord = camera.location + (camera.right * screen_offset.x - camera.up * screen_offset.y);
}

Ray RayIterator::operator*() const {
    return camera.project(world_cord);
}

RayIterator RayIterator::operator++(int) {
    RayIterator tmp(*this);
    this->operator++();
    return tmp;
}

RayIterator& RayIterator::operator++() {
    if(screen_cord.x == camera.resolution.x) {
        screen_cord = Vector2f(0, screen_cord.y + 1);
        // Move all the way back left, then down 1
        world_cord = world_cord - (camera.right * camera.resolution.x) - camera.up;
    } else {
        screen_cord.x += 1;
        world_cord = world_cord + camera.right;
    }

    Vector2f screen_offset = (screen_cord - (camera.resolution / 2));
    //camera.resolution.print();
    // Minus y here since screen coords up is -y
    world_cord = camera.location + (camera.right * screen_offset.x - camera.up * screen_offset.y);

    return *this;
};

bool RayIterator::operator==(const RayIterator& other) {
    // Need to check for different cameras but will ignore for now
    return this->screen_cord == other.screen_cord;
}

bool RayIterator::operator!=(const RayIterator& other) {
    return !(*this == other);
}

OrthoCamera::OrthoCamera(Vector3f location_, Vector3f up_, Vector3f right_,
        Vector2f size_, Vector2f resolution_) :
    Camera(location_, up_, right_, size_, resolution_) {}

Ray OrthoCamera::project(Vector3f world_cord) {
    return Ray(world_cord, norm);
}

PerspectiveCamera::PerspectiveCamera(Vector3f location_, Vector3f up_, Vector3f right_,
        Vector2f size_, Vector2f resolution_, Vector3f eye_) :
    Camera(location_, up_, right_, size_, resolution_), eye(eye_) {}

Ray PerspectiveCamera::project(Vector3f world_cord) {
    Ray r;
    r.fromPoints(eye, world_cord);
    //r.origin = world_cord;
    r.direction.normalize();
    return r;
}

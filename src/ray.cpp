#include "ray.h"


void Ray::fromPoints(Vector3f start, Vector3f end) {
    this->origin = start;
    //this->direction = (start - end).normalize();
    this->direction = (end - start).normalize();
}

Vector3f Ray::getPoint(float t) {
    return origin + direction * t;
}

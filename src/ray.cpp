#include "ray.h"


void Ray::fromPoints(Vector3f start, Vector3f end) {
    this->origin = start;
    this->direction = (end - start).normalize();
}

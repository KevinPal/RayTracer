#include "ray.h"
#include <stdio.h>


// Creates a ray between two points
void Ray::fromPoints(Vector3f start, Vector3f end) {
    this->origin = start;
    this->direction = (end - start);
}

// Gets a point along the paramateric equation for this ray
Vector3f Ray::getPoint(float t) {
    return origin + direction * t;
}

// Prints this ray
void Ray::print() {
    printf("O: <%f %f %f> D: <%f %f %f>\n", origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
}

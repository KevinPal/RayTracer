#include "ray.h"
#include <stdio.h>


void Ray::fromPoints(Vector3f start, Vector3f end) {
    this->origin = start;
    //this->direction = (end - start).normalize();
    this->direction = (end - start);
}

Vector3f Ray::getPoint(float t) {
    return origin + direction * t;
}

void Ray::print() {
    printf("O: <%f %f %f> D: <%f %f %f>\n", origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
}

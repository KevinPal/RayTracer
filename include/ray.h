#ifndef RAY_H
#define RAY_H

#include "vector.h"

class Ray {

    public:
        Vector3f origin;
        Vector3f direction;

        Ray() : origin(0, 0), direction(1, 0) {}
        Ray(Vector3f origin_, Vector3f direction_) : origin(origin_), direction(direction_) {}

        void fromPoints(Vector3f start, Vector3f end);
        Vector3f getPoint(float t);
        void print();

};

#endif

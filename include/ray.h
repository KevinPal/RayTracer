#ifndef RAY_H
#define RAY_H

#include "vector.h"

/*
 * Defines a ray with origin and direction
 */
class Ray {

    public:
        Vector3f origin;
        Vector3f direction;

        // Default ray from the origin along the x axis
        Ray() : origin(0, 0), direction(1, 0) {}

        // Generic ray with given origin and direction
        Ray(Vector3f origin_, Vector3f direction_) : origin(origin_), direction(direction_) {}

        // Recreates this ray starting from "start" and passing through end at t=1
        void fromPoints(Vector3f start, Vector3f end);

        // Gets a point along this ray
        Vector3f getPoint(float t);

        // Prints the ray
        void print();

};

#endif

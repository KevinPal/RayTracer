#ifndef RAY_H
#define RAY_H

#include "vector.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

/*
 * Defines a ray with origin and direction
 */
class Ray {

    public:
        Vector3f origin;
        Vector3f direction;

        // Default ray from the origin along the x axis
        __device__ __host__ Ray() : origin(0, 0), direction(1, 0) {}

        // Generic ray with given origin and direction
        __device__ __host__ Ray(Vector3f origin_, Vector3f direction_) : origin(origin_), direction(direction_) {}

        // Recreates this ray starting from "start" and passing through end at t=1
        __device__ __host__ void fromPoints(Vector3f start, Vector3f end);

        // Gets a point along this ray
        __device__ __host__ Vector3f getPoint(float t);

        // Prints the ray
        __device__ __host__ void print();

};

#endif

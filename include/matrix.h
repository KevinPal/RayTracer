#ifndef MATRIX44_H
#define MATRIX44_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "vector.h"

class Matrix44 {

    public:
        float data[16];

        Matrix44(float data[16]);

        __host__ __device__ Vector3f transform(Vector3f v);
        Matrix44 invert();
        Matrix44 transpose();

        void print();

        
};

#endif

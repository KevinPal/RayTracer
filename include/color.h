#ifndef COLOR_H
#define COLOR_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#define HOSTDEVICE __host__ __device__

#include "vector.h"

/* A class to represent colors. 
 * For now, this is simply a vector3f that should
 * have compoeneted each on [0, 1]
 */
class Color : public Vector3f{
    public:

        // Creates the color black
        HOSTDEVICE Color() : Vector3f() {};

        // Copy constuctor from another color
        HOSTDEVICE Color(const Color& other):
            Vector3f(other) {}

        // Specify the color from another vector
        HOSTDEVICE Color(const Vector3f& other):
            Vector3f(other) {}

        // Specify the B G R of the color directly
        HOSTDEVICE Color(float x_, float y_, float z_) :
            Vector3f(x_, y_, z_) {}

        // Writes the color to a buffer in the BGRA format
        HOSTDEVICE unsigned char* writeToBuff(unsigned char* buff);

        // Clamps the components of the color
        HOSTDEVICE void clamp(void);
};

#endif

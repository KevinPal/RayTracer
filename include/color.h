#ifndef COLOR_H
#define COLOR_H

#include "vector.h"

/* A class to represent colors. 
 * For now, this is simply a vector3f that should
 * have compoeneted each on [0, 1]
 */
class Color : public Vector3f{
    public:

        // Creates the color black
        Color() : Vector3f() {};

        // Copy constuctor from another color
        Color(const Color& other):
            Vector3f(other) {}

        // Specify the color from another vector
        Color(const Vector3f& other):
            Vector3f(other) {}

        // Specify the B G R of the color directly
        Color(float x_, float y_, float z_) :
            Vector3f(x_, y_, z_) {}

        // Writes the color to a buffer in the BGRA format
        unsigned char* writeToBuff(unsigned char* buff);

        // Clamps the components of the color
        void clamp(void);
};

#endif

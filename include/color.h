#ifndef COLOR_H
#define COLOR_H

#include "vector.h"

class Color : public Vector3f{
    public:

        Color() : Vector3f() {};

        Color(const Color& other):
            Vector3f(other) {}

        Color(const Vector3f& other):
            Vector3f(other) {}

        Color(float x_, float y_, float z_) :
            Vector3f(x_, y_, z_) {}

        unsigned char* writeToBuff(unsigned char* buff);
        void clamp(void);
};

#endif

#ifndef COLOR_H
#define COLOR_H

#include "vector.h"

class Color {
    public:
        Vector3f color;
        float alpha;
        float specular;

        Color();
        Color(Vector3f color_, float alpha_, float specular_);
        Color(int r, int g, int b, float a, float spec);
        Color(int r, int g, int b, float a);
        Color(int r, int g, int b);
        Color(Vector3f color);
        Color(Vector3f color, float alpha);
        Color(const Color& other);

        unsigned char* writeToBuff(unsigned char* buff);
        void clamp(void);
};

#endif

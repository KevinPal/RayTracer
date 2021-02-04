#ifndef COLOR_H
#define COLOR_H

#include "vector.h"

class Color {
    public:
        Vector3f color;
        int alpha;

        Color();
        Color(int r, int g, int b, int a);
        Color(int r, int g, int b);
        Color(Vector3f color);
        Color(Vector3f color, int alpha);

        unsigned char* writeToBuff(unsigned char* buff);
        void clamp(void);
};

#endif

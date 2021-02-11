#include "color.h"
#include "vector.h"
#include <stdio.h>

// Default constructor to green
Color::Color(int r, int g, int b, float a, float spec) :
    color(Vector3f(r, g, b)), alpha(a), specular(spec) {}

Color::Color(int r, int g, int b, float a) :
    color(Vector3f(r, g, b)), alpha(a), specular(0) {}

Color::Color(int r, int g, int b) :
    Color(r, g, b, 1, 0) {}

Color::Color(Vector3f color_, float alpha_, float specular_) :
    color(color_), alpha(alpha_), specular(specular_) { }

Color::Color() :
    Color(Vector3f(0, 255, 0), 1, 0) {}

Color::Color(Vector3f color_) :
    Color(color_, 1, 0) {}



Color::Color(const Color& other):
    Color(other.color, other.alpha, other.specular) {}

unsigned char* Color::writeToBuff(unsigned char* buff) {
    buff[0] = this->color.x;
    buff[1] = this->color.y;
    buff[2] = this->color.z;
    buff[3] = (int) (alpha * 255);
    return buff + 4;
}

void Color::clamp(void) {
    if(color.x < 0)
        color.x = 0;
    if(color.x > 255)
        color.x = 255;
    if(color.y < 0)
        color.y = 0;
    if(color.y > 255)
        color.y = 255;
    if(color.z < 0)
        color.z = 0;
    if(color.z > 255)
        color.z = 255;
}

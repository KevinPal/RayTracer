#include "color.h"
#include "vector.h"

// Default constructor to green
Color::Color(int r, int g, int b, int a) :
    color(Vector3f(r, g, b)), alpha(a) {}

Color::Color(int r, int g, int b) :
    Color(r, g, b, 255) {}

Color::Color() :
    Color(Vector3f(0, 255, 0), 255) {}

Color::Color(Vector3f color_) :
    Color(color_, 255) {}

Color::Color(Vector3f color_, int alpha_) :
    color(color_), alpha(alpha_) {}

unsigned char* Color::writeToBuff(unsigned char* buff) {
    buff[0] = this->color.x;
    buff[1] = this->color.y;
    buff[2] = this->color.z;
    buff[3] = alpha;
    return buff + 4;
}

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

Color::Color(const Color& other):
    Color(other.color, other.alpha) {}

unsigned char* Color::writeToBuff(unsigned char* buff) {
    buff[0] = this->color.x;
    buff[1] = this->color.y;
    buff[2] = this->color.z;
    buff[3] = alpha;
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

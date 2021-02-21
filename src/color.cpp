#include "color.h"
#include "vector.h"
#include <stdio.h>

unsigned char* Color::writeToBuff(unsigned char* buff) {
    buff[0] = (char) (this->x * 255);
    buff[1] = (char) (this->y * 255);
    buff[2] = (char) (this->z * 255);
    buff[3] = 0;
    return buff + 4;
}

void Color::clamp(void) {
    if(x < 0)
        x = 0;
    if(x > 1)
        x = 1;
    if(y < 0)
        y = 0;
    if(y > 1)
        y = 1;
    if(z < 0)
        z = 0;
    if(z > 1)
        z = 1;
}

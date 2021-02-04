#include <stdio.h>
#include "display.h"

#include "plane.h"


int main (int argc, char **argv) {

    Display* display = Display::getInstance();

    display->init(500, 500);

    unsigned char* buf = display->getBuffer();

    Plane p(Vector3f(0, 0, -5), Vector3f(0, 1, -1), Color(255, 0, 0));

    IntersectData data;
    Color bg;
    for(int x = 0; x < display->getWidth(); x++) {
        for(int y = 0; y < display->getHeight(); y++) {

            Ray r(Vector3f(x - display->getWidth()/2, (y - display->getHeight()/2), 0), Vector3f(0, 0, 1));

            data = p.intersects(r);
        
            (data.t < 0 ? data.color : bg).writeToBuff(&(buf[4 * (y * display->getWidth() + x)]));
        }
    }


    display->run();
    display->destory();

  return 0;
}

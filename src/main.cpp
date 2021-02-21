#include <stdio.h>

#include "primitives.h"
#include "mesh.h"
#include "camera.h"
#include "material.h"
#include "display.h"
#include "renderer.h"

#include <vector>
#include <math.h>
#include <cassert>
#include <math.h>
#include <stack>

int main (int argc, char **argv) {


    Display* display = Display::getInstance();

    display->init(1500, 1500);

    unsigned char* buf = display->getBuffer();

    Plane p(
         Vector3f(0, 0, 15),
         Vector3f(0, 0, -1),
         Material(Color(0.0f, 0.0f, 1.0f), 1, 0, 0));

    Plane p2(
         Vector3f(0, 0, 0),
         Vector3f(0, 1, 0),
         Material(Color(1.0f, 0.0f, 0.0f), 1, 0, 0.25));

    Sphere s( 
         Vector3f(0, 5, 5), 
         5,
         Material(Color(1.0f, 0.0f, 1.0f), 1, 0, 0));

    Triangle t(
            Vector3f(-15, 10, 10),
            Vector3f(0, 20, 0),
            Vector3f(15, 10, 10),
            Material(Color(1.0f, 1.0f, 0.0f), 1, 0, 1));

    Prism r(
            Vector3f(5, 0, -5),
            Vector3f(0, 1, 0),
            Vector3f(1, 0, 1),
            Vector3f(5, 10, 5),
            Material(Color(0.0f, 1.0f, 1.0f), 0.5, 0, 0));

    Prism r2(
            Vector3f(-7, 1, -6),
            Vector3f(0, 1, 0),
            Vector3f(0, 1, 2),
            Vector3f(5, 5, 5),
            Material(Color(.5f, .5f, .5f), 1, 0, 0));

    Mesh scene;
    scene.addObject(&p);
    scene.addObject(&p2);
    scene.addObject(&s);
    scene.addObject(&t);
    scene.addObject(&r);
    scene.addObject(&r2);


    IntersectData data;
    Color bg(100, 100, 100);

    PerspectiveCamera cam = PerspectiveCamera(
        Vector3f(0, 5, -5),
        Vector3f(0, 1.0, 0),
        Vector3f(1.0, 0, 0),
        Vector2f(25.0, 25.0),
        Vector2f(display->getWidth(), display->getHeight()),
        Vector3f(0, 5, -25)
    );
    
    /*
    OrthoCamera cam = OrthoCamera(
        Vector3f(0, 5, -25),
        Vector3f(0, 1.0, .1),
        Vector3f(1.0, 0, 0),
        Vector2f(50, 50),
        Vector2f(display->getWidth(), display->getHeight())
    );
    */


    for (auto it = cam.begin(), end = cam.end(); it != end; ++it) {
        Ray ray = *it;
        Vector2f screen_coord = it.getScreenCord();

        IntersectData hit = renderRay(ray, &scene, 3);

        (hit.t >= 0 ? hit.material.color : bg).writeToBuff(&(buf[(int)(4 * (screen_coord.y * display->getWidth() + screen_coord.x))]));
    }

    display->run();
    display->destory();

  return 0;
}

#include <stdio.h>

#include "primitives.h"
#include "mesh.h"
#include "camera.h"
#include "material.h"
#include "display.h"
#include "renderer.h"
#include "antialias.h"
#include "BVH.h"

#include <vector>
#include <math.h>
#include <cassert>
#include <math.h>
#include <stack>
#include <ctime>

#define DO_ANTI_ALIASING 0
#define DO_BVH 0
#define BVH_LEAF_SIZE 2
#define CAMERA_NUM 4

#define RAND ((float) (rand() / (float) RAND_MAX))

int main (int argc, char **argv) {


    // Setup the display and get the buffer
    Display* display = Display::getInstance();
    display->init(750, 750);
    unsigned char* buf = display->getBuffer();

    BVHNode scene(BVH_LEAF_SIZE);

    /*
    Plane p(
         Vector3f(0, 0, 15),
         Vector3f(0, 0, -1),
         Material(Color(0.0f, 0.0f, 1.0f), 1, 0, 0));

    Plane p2(
         Vector3f(0, 0, 0),
         Vector3f(0, 1, 0),
         Material(Color(1.0f, 0.0f, 0.0f), 1, 0, 0.75));

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
            Material(Color(0.0f, 1.0f, 1.0f), 0.25, 0, 0));

    Prism r2(
            Vector3f(-7, 1, -6),
            Vector3f(0, 1, 0),
            Vector3f(0, 1, 2),
            Vector3f(5, 5, 5),
            Material(Color(.5f, .5f, .5f), 1, 0, 0));

    scene.addObject(&p);
    scene.addObject(&p2);
    scene.addObject(&s);
    scene.addObject(&t);
    scene.addObject(&r);
    scene.addObject(&r2);
    */

    int num_spheres = 0;
    while(num_spheres < 1000) {
        float sx = RAND * 20 - 10;
        float sy = RAND * 20 - 10;
        float sz = RAND * 100;

        float r = RAND * 0.5 + 0.5;
        float g = RAND * 0.5 + 0.5;
        float b = RAND * 0.5 + 0.5;

        float rad = RAND * 0.5 + 0.1;

        Sphere* s = new Sphere( 
            Vector3f(sx, sy + 5, sz), 
            rad,
            Material(Color(r, g, b), 1, 0, RAND)
        );
        scene.addObject(s);
        num_spheres += 1;
    }


    printf("Num spheres: %d\n", num_spheres);

    std::time_t build_start = std::time(NULL);
    scene.partition();
    printf("Parition time: %d\n", std::time(NULL) - build_start);

    //scene.addObject(scene.left->left->bounding_box);
    //scene.addObject(scene.right->left->right->bounding_box);
    //scene.addObject(scene.right->left->left->right->bounding_box);

    //scene.objects[scene.objects.size()-1]->material.color[0] = 255.0;
    //scene.objects[scene.objects.size()-1]->material.color[1] = 255.0;
    //scene.objects[scene.objects.size()-1]->material.color[2] = 255.0;
    //scene.addObject(scene.right->bounding_box);

    IntersectData data;
    Color bg(100, 100, 100);


    // Specify the camera. Currently hard coding
    // until we get something more fancy

    Camera* cam;
    int cam_num = CAMERA_NUM;
    
    switch(cam_num) {
        case 1:
            cam = new PerspectiveCamera(
                Vector3f(0, 5, -5),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 0),
                Vector2f(25.0, 25.0),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(0, 5, -25)
            );
            break;
        case 2:
            cam = new PerspectiveCamera(
                Vector3f(-10, 5, -5),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, -1),
                Vector2f(15.0, 15.0),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(-30, 5, -25)
            );
            break;
        case 3:
            // Other angle
            cam = new PerspectiveCamera(
                Vector3f(10, 5, -5),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 1),
                Vector2f(25.0, 25.0),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(30, 5, -25)
            );
            break;
        case 4:
            cam = new OrthoCamera(
                Vector3f(0, 5, -25),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 0),
                Vector2f(25, 25),
                Vector2f(display->getWidth(), display->getHeight())
            );
    }

    std::time_t start = std::time(NULL);

    // Loop across all the rays generated by the camera
    int i = 0;
    for (auto it = cam->begin(), end = cam->end(); it != end; ++it) {
        Ray ray = *it;
        Vector2f screen_coord = it.getScreenCord();
        Vector3f color = Vector3f(0, 0, 0);

        AntiAliaser* anti_aliaser;
        int count = 0;

        // If we have an anti aliaser, loop across all the rays from it, and do an intersection test
        // Otherwise just do it for the main ray
        if(DO_ANTI_ALIASING) {
            for(anti_aliaser = new GridAntiAliaser(&it, 2); !anti_aliaser->isDone(); ++(*anti_aliaser)) {
                IntersectData hit = renderRay(**anti_aliaser, &scene, 3);
                count += 1;
                color = color + (hit.t >= 0 ? hit.material.color : bg);
            }
        } else {
            IntersectData hit = renderRay(ray, &scene, 2);
            count += 1;
            color = color + (hit.t >= 0 ? hit.material.color : bg);
        }

        // Average the colors and write to the buffer
        color = color / count;
        Color(color).writeToBuff(&(buf[(int)(4 * (screen_coord.y * display->getWidth() + screen_coord.x))]));

        // Print progress
        i += 1;
        //if((i * 100 / (display->getWidth() * display->getHeight())) % 5 == 0)
            printf("%.3f\%\n",  i * 100.0 / (display->getWidth() * display->getHeight()));
    }

    std::time_t end = std::time(NULL);
    printf("Took %d\n", end - start);

    // Write to a PNG for handing
    //display->writeToPNG("images/mp1/anti_aliased.png");

    // Display the image
    display->run();
    display->destory();

  return 0;
}

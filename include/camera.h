#ifndef CAMERA_H
#define CAMERA_H

#include "vector.h"
#include "ray.h"
#include <iterator>

class Camera;

class RayIterator {

    private:
        Camera& camera;
        Vector3f world_cord;
        Vector2f screen_cord;

    public:
        RayIterator(Camera& camera, int x, int y);

        Ray operator*() const;
        RayIterator& operator++();
        RayIterator operator++(int);

        Vector2f getScreenCord() { return screen_cord; }
        Vector3f getWorldCord() { return world_cord; }

        bool operator==(const RayIterator& other);
        bool operator!=(const RayIterator& other);
        
};

class Camera {

    friend RayIterator;

    public:
        Vector3f location;
        Vector3f up;
        Vector3f right;
        Vector3f norm;

        Vector2f size;
        Vector2f resolution;


        Camera(Vector3f location_, Vector3f up_, Vector3f right_,
                Vector2f size_, Vector2f resolution_);

        virtual Ray project(Vector3f world_cord) = 0;

        RayIterator begin() { return RayIterator(*this, 0, 0); }
        RayIterator end() { return RayIterator(*this, resolution.x, resolution.y); }
};

class OrthoCamera : public Camera {

    public:
        OrthoCamera(Vector3f location_, Vector3f up_, Vector3f right_,
                Vector2f size_, Vector2f resolution_);

        Ray project(Vector3f world_cord);

};

class PerspectiveCamera : public Camera {

    private:
        Vector3f eye;

    public:
        PerspectiveCamera(Vector3f location_, Vector3f up_, Vector3f right_,
                Vector2f size_, Vector2f resolution_, Vector3f eye);

        Ray project(Vector3f world_cord);

};

#endif

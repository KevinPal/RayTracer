#ifndef CAMERA_H
#define CAMERA_H

#include "vector.h"
#include "ray.h"
#include <iterator>

class Camera;

/*
 * Iterator that a Camera class provides. Uses the project() method
 * of the underlying camera to generate a ray for each pixel on the 
 * screen.
 */
class RayIterator {

    private:
        Vector3f world_cord;
        Vector2f screen_cord;

    public:
        Camera& camera;

        // Constructor. Creates an iterator, and starts rendering the pixel at the given x,y
        RayIterator(Camera& camera, int x, int y);

        // Gets the ray that the iterator currently points to. Uses the project function
        // of the underlying camera
        Ray operator*() const;

        // Advanced the iterator to the next pixel
        RayIterator& operator++();
        RayIterator operator++(int);

        // Gets the current screen coordinates of the ray iterator
        Vector2f getScreenCord() const { return screen_cord; }
        // Gets the current world coordinates of the current ray iterator
        Vector3f getWorldCord() const { return world_cord; }

        // Comparison operators to check if two ray iterators are the same
        bool operator==(const RayIterator& other);
        bool operator!=(const RayIterator& other);
        
};

/**
 * Generic Camera abstract class. Each camera is defined by a 
 * view plane, the resolution of the view plane, a location in the world,
 * and an up and right vector. Iteration over the camera provides rays 
 * for every pixel in the view plane. Subclasses should override the
 * project() method for various camera types
 */
class Camera {

    friend RayIterator;

    public:
        Vector3f location;
        Vector3f up;
        Vector3f right;
        Vector3f norm;

        Vector2f size;
        Vector2f resolution;

        // Camera constructor, specifying all the elements of a generic camera
        Camera(Vector3f location_, Vector3f up_, Vector3f right_,
                Vector2f size_, Vector2f resolution_);

        // Method to be implemented by subclasses to define how to project a ray to
        // a given world coordinate
        virtual Ray project(Vector3f world_cord) = 0;

        // Iterator methods
        RayIterator begin() { return RayIterator(*this, 0, 0); }
        RayIterator end() { return RayIterator(*this, resolution.x, resolution.y); }
};

/*
 * The orthographic camera sends rays perpendicular to the given up and right vectors
 */
class OrthoCamera : public Camera {

    public:
        OrthoCamera(Vector3f location_, Vector3f up_, Vector3f right_,
                Vector2f size_, Vector2f resolution_);

        Ray project(Vector3f world_cord);

};

/*
 * The perpective camera sends rays out from a specified eye coordinate through
 * the world coordinates of the pixel
 */
class PerspectiveCamera : public Camera {

    private:
        Vector3f eye;

    public:
        PerspectiveCamera(Vector3f location_, Vector3f up_, Vector3f right_,
                Vector2f size_, Vector2f resolution_, Vector3f eye);

        Ray project(Vector3f world_cord);

};

#endif

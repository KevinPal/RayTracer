#ifndef ANTIALIAS_H
#define ANTIALIAS_H

#include "camera.h"
#include "ray.h"
#include "vector.h"

class AntiAliaser {

    public:
        const RayIterator* const ray_iter;

        AntiAliaser(const RayIterator* iter):
            ray_iter(iter) {}

        virtual Ray operator*() const = 0;
        virtual AntiAliaser& operator++() = 0;
        virtual bool isDone() = 0;


};

class GridAntiAliaser : public AntiAliaser {

    private:
        Vector2f world_cord;
        Vector2f screen_offset;
        Ray primary_ray;


    public:
        int grid_size;

        GridAntiAliaser(const RayIterator* iter, int grid_size_);

        Ray operator*() const override;
        GridAntiAliaser& operator++() override;
        bool isDone() override;

};

#endif

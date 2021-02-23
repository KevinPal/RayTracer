#ifndef ANTIALIAS_H
#define ANTIALIAS_H

#include "camera.h"
#include "ray.h"
#include "vector.h"

/**
 * Generic Antialsier interface. Wraps around
 * a ray interator the generate subrays. Generation
 * methods should be subclasses of this class
 */
class AntiAliaser {

    public:
        const RayIterator* const ray_iter;

        // Constructor, create an iterator for this ray
        AntiAliaser(const RayIterator* iter):
            ray_iter(iter) {}

        // Get the subray that this iterator currently points to
        virtual Ray operator*() const = 0;

        // Advance to the next subray
        virtual AntiAliaser& operator++() = 0;

        // Check if this iterator is done generating subrays
        virtual bool isDone() = 0;


};


/*
 * An implementation of an antialiasing method
 * Generates subrays in an even grid
 */
class GridAntiAliaser : public AntiAliaser {

    private:
        Vector2f world_cord;
        Vector2f screen_offset;
        Ray primary_ray;


    public:
        int grid_size;

        // Wraps a ray iterator. Will generate grid_size * grid_size
        // subrays in a uniform way
        GridAntiAliaser(const RayIterator* iter, int grid_size_);

        Ray operator*() const override;
        GridAntiAliaser& operator++() override;
        bool isDone() override;

};

#endif

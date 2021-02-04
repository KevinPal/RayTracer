#ifndef VECTOR_H
#define VECTOR_H

class Vector2f;
class Vector3f;

class Vector2f {
    public:
        float x;
        float y;

        Vector2f(const Vector2f& other);
        Vector2f(const Vector3f& other);
        Vector2f(float x_, float y_);

        float length();
        Vector2f normalize();

        /*
        static Vector2f subtract(const Vector2f& v1, const Vector2f& v2);
        static Vector2f add(const Vector2f& v1, const Vector2f& v2);
        */
};

class Vector3f : public Vector2f {
    public:
        float z;

        Vector3f(const Vector3f& other);
        Vector3f(const Vector2f& other);
        Vector3f(float x_, float y_);
        Vector3f(float x_, float y_, float z_);

        float length() const;
        Vector3f normalize();

        Vector3f operator-(const Vector3f& other) const;
        Vector3f operator+(const Vector3f& other) const;
        float dot(const Vector3f& other) const;

};

#endif

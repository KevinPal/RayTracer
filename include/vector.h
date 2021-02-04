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
        Vector2f();

        float length();
        Vector2f normalize();

        Vector2f operator-(const Vector2f& other) const;
        Vector2f operator+(const Vector2f& other) const;
        Vector2f operator/(const Vector2f& other) const;
        Vector2f operator*(const Vector2f& other) const;

        Vector2f operator/(const float& other) const;
        Vector2f operator*(const float& other) const;

        bool operator==(const Vector2f& other) const;
        bool operator!=(const Vector2f& other) const;

        void print(void) const;
};

class Vector3f {

    public:
        float x;
        float y;
        float z;

        Vector3f(const Vector3f& other);
        Vector3f(const Vector2f& other);
        Vector3f();
        Vector3f(float x_, float y_);
        Vector3f(float x_, float y_, float z_);

        float length() const;
        Vector3f normalize();

        Vector3f operator-(const Vector3f& other) const;
        Vector3f operator+(const Vector3f& other) const;

        Vector3f operator/(const Vector3f& other) const;
        Vector3f operator*(const Vector3f& other) const;

        Vector3f operator/(const float& other) const;
        Vector3f operator*(const float& other) const;

        float dot(const Vector3f& other) const;
        Vector3f cross(const Vector3f& other) const;

        void print(void) const;
};

#endif

#ifndef CAMERA_H
#define CAMERA_H

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum CameraMovement { FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN };

struct BoundingBox {
  glm::vec3 min{FLT_MAX};
  glm::vec3 max{-FLT_MAX};

  glm::vec3 center() const { return (min + max) * 0.5f; }
  glm::vec3 size() const { return max - min; }
  float radius() const { return glm::length(size()) * 0.5f; }
  bool isValid() const { return min.x <= max.x; }
};

class Camera {
public:
  glm::vec3 Position;
  glm::vec3 Front;
  glm::vec3 Up;
  glm::vec3 Right;
  glm::vec3 WorldUp;

  float Yaw;
  float Pitch;
  float Zoom;

  // Configurable speeds (exposed to ImGui)
  float MovementSpeed = 2.5f;
  float MouseSensitivity = 0.1f;
  float ScrollSpeed = 1.0f;
  float OrbitSpeed = 0.3f;

  // Orbit camera state
  glm::vec3 Pivot{0.0f};
  float OrbitDistance = 5.0f;

  // Model-relative scaling
  float ModelRadius = 1.0f;

  Camera(glm::vec3 position = glm::vec3(0.0f, 1.0f, 3.0f),
         glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f,
         float pitch = 0.0f);

  glm::mat4 getViewMatrix() const;
  void processKeyboard(CameraMovement direction, float deltaTime);
  void processMouseMovement(float xoffset, float yoffset,
                            bool constrainPitch = true);
  void processMouseScroll(float yoffset, float deltaTime);
  void orbit(float deltaX, float deltaY, float deltaTime);
  void pan(float deltaX, float deltaY, float deltaTime);

  // Auto-framing based on model bounds
  void frameModel(const BoundingBox &bounds, float distanceMultiplier = 2.5f);
  void reset(const BoundingBox &bounds);

  // Getters for projection setup
  float getNearPlane() const;
  float getFarPlane() const;

private:
  void updateCameraVectors();
  void updatePositionFromOrbit();
};

#endif

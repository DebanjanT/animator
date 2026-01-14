#include "camera.h"
#include <algorithm>
#include <cfloat>

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : Front(glm::vec3(0.0f, 0.0f, -1.0f)), Zoom(45.0f) {
  Position = position;
  WorldUp = up;
  Yaw = yaw;
  Pitch = pitch;
  updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const {
  return glm::lookAt(Position, Pivot, Up);
}

void Camera::processKeyboard(CameraMovement direction, float deltaTime) {
  // Scale movement by model radius for consistent feel across models
  float velocity = MovementSpeed * ModelRadius * deltaTime;

  if (direction == FORWARD) {
    Position += Front * velocity;
    Pivot += Front * velocity;
  }
  if (direction == BACKWARD) {
    Position -= Front * velocity;
    Pivot -= Front * velocity;
  }
  if (direction == LEFT) {
    Position -= Right * velocity;
    Pivot -= Right * velocity;
  }
  if (direction == RIGHT) {
    Position += Right * velocity;
    Pivot += Right * velocity;
  }
  if (direction == UP) {
    Position += WorldUp * velocity;
    Pivot += WorldUp * velocity;
  }
  if (direction == DOWN) {
    Position -= WorldUp * velocity;
    Pivot -= WorldUp * velocity;
  }
}

void Camera::processMouseMovement(float xoffset, float yoffset,
                                  bool constrainPitch) {
  xoffset *= MouseSensitivity;
  yoffset *= MouseSensitivity;

  Yaw += xoffset;
  Pitch += yoffset;

  if (constrainPitch) {
    Pitch = std::clamp(Pitch, -89.0f, 89.0f);
  }

  updateCameraVectors();
  updatePositionFromOrbit();
}

void Camera::processMouseScroll(float yoffset, float deltaTime) {
  // Zoom by adjusting orbit distance, scaled by model radius
  float zoomAmount = yoffset * ScrollSpeed * ModelRadius * deltaTime * 10.0f;
  OrbitDistance -= zoomAmount;
  OrbitDistance = std::max(OrbitDistance, ModelRadius * 0.1f);
  OrbitDistance = std::min(OrbitDistance, ModelRadius * 100.0f);
  updatePositionFromOrbit();
}

void Camera::orbit(float deltaX, float deltaY, float deltaTime) {
  // Orbit around pivot point
  Yaw += deltaX * OrbitSpeed;
  Pitch -= deltaY * OrbitSpeed;

  // Clamp pitch to avoid gimbal lock
  Pitch = std::clamp(Pitch, -89.0f, 89.0f);

  updateCameraVectors();
  updatePositionFromOrbit();
}

void Camera::pan(float deltaX, float deltaY, float deltaTime) {
  // Pan camera and pivot together
  float panSpeed = MovementSpeed * ModelRadius * 0.01f;
  glm::vec3 offset = Right * (-deltaX * panSpeed) + Up * (deltaY * panSpeed);
  Position += offset;
  Pivot += offset;
}

void Camera::frameModel(const BoundingBox &bounds, float distanceMultiplier) {
  if (!bounds.isValid())
    return;

  Pivot = bounds.center();
  ModelRadius = bounds.radius();

  // Calculate distance to fit model in view
  float fovRad = glm::radians(Zoom);
  OrbitDistance = (ModelRadius / sin(fovRad * 0.5f)) * distanceMultiplier;

  // Reset orientation
  Yaw = -90.0f;
  Pitch = 15.0f; // Slight downward angle

  updateCameraVectors();
  updatePositionFromOrbit();
}

void Camera::reset(const BoundingBox &bounds) { frameModel(bounds, 2.5f); }

float Camera::getNearPlane() const {
  return std::max(0.001f, ModelRadius * 0.01f);
}

float Camera::getFarPlane() const { return ModelRadius * 200.0f; }

void Camera::updateCameraVectors() {
  glm::vec3 front;
  front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
  front.y = sin(glm::radians(Pitch));
  front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
  Front = glm::normalize(front);
  Right = glm::normalize(glm::cross(Front, WorldUp));
  Up = glm::normalize(glm::cross(Right, Front));
}

void Camera::updatePositionFromOrbit() {
  // Position camera at OrbitDistance from Pivot based on Yaw/Pitch
  Position = Pivot - Front * OrbitDistance;
}

#ifndef VIEWER_H
#define VIEWER_H

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "animator.h"
#include "camera.h"
#include "model.h"
#include "shader.h"

class MoCapViewer {
public:
  MoCapViewer(int width = 1280, int height = 720,
              const std::string &title = "MoCap Viewer");
  ~MoCapViewer();

  bool initialize();
  void loadModel(const std::string &path);
  void run();
  void runAsync();
  bool runOneFrame();  // Process single frame, returns false if should quit
  void stop();
  bool isRunning() const { return running; }
  bool shouldClose() const;

  // Animation control
  void setBoneTransform(const std::string &boneName, float px, float py,
                        float pz, float qw, float qx, float qy, float qz,
                        float sx, float sy, float sz);
  void setBoneMatrices(const std::vector<float> &matrices);
  void setAnimationFrame(
      const std::map<std::string, std::vector<float>> &boneTransforms);

  // Camera control
  void setCameraPosition(float x, float y, float z);
  void setCameraTarget(float x, float y, float z);
  void setBackgroundColor(float r, float g, float b);
  void resetCamera();

  // Getters
  int getWidth() const { return width; }
  int getHeight() const { return height; }

private:
  int width, height;
  std::string title;
  GLFWwindow *window;

  std::unique_ptr<Shader> shader;
  std::unique_ptr<Model> model;
  std::unique_ptr<Animator> animator;
  std::unique_ptr<Camera> camera;

  glm::vec3 backgroundColor;
  std::atomic<bool> running;
  std::mutex dataMutex;

  float lastX, lastY;
  bool firstMouse;
  float deltaTime, lastFrame;

  // Model bounds for camera auto-framing
  BoundingBox modelBounds;
  std::string modelPath;

  // UI state
  bool showGrid;
  bool showImGuiDemo;
  bool showGizmo;
  
  // Gizmo state
  int gizmoOperation;  // ImGuizmo::OPERATION
  int gizmoMode;       // ImGuizmo::MODE
  glm::mat4 modelMatrix;
  bool useSnap;
  float snapTranslation[3];
  float snapRotation;
  float snapScale;
  
  // Ground plane
  unsigned int groundPlaneVAO, groundPlaneVBO;
  void setupGroundPlane();
  void renderGroundPlane();

  void processInput();
  void render();
  void renderImGui();
  void renderBoneHierarchy(const BoneNode &node);

  static void framebufferSizeCallback(GLFWwindow *window, int width,
                                      int height);
  static void mouseCallback(GLFWwindow *window, double xpos, double ypos);
  static void scrollCallback(GLFWwindow *window, double xoffset,
                             double yoffset);

  void setupShaders();
  void setupImGui();
  void drawGrid();
};

#endif

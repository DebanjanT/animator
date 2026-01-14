#include "viewer.h"
#include <cfloat>
#include <iostream>
#include <thread>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

static MoCapViewer *currentViewer = nullptr;

MoCapViewer::MoCapViewer(int width, int height, const std::string &title)
    : width(width), height(height), title(title), window(nullptr),
      backgroundColor(0.2f, 0.2f, 0.25f), running(false), lastX(width / 2.0f),
      lastY(height / 2.0f), firstMouse(true), deltaTime(0.0f), lastFrame(0.0f),
      showGrid(true), showImGuiDemo(false) {}

MoCapViewer::~MoCapViewer() {
  stop();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  if (window) {
    glfwDestroyWindow(window);
  }
  glfwTerminate();
}

bool MoCapViewer::initialize() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSetCursorPosCallback(window, mouseCallback);
  glfwSetScrollCallback(window, scrollCallback);

  currentViewer = this;

  int version = gladLoadGL(glfwGetProcAddress);
  if (version == 0) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return false;
  }

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f));
  animator = std::make_unique<Animator>();

  setupShaders();
  setupImGui();

  std::cout << "OpenGL Viewer initialized" << std::endl;
  std::cout << "  OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

  return true;
}

void MoCapViewer::setupImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
}

void MoCapViewer::setupShaders() {
  const char *vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoords;
        layout (location = 5) in ivec4 aBoneIds;
        layout (location = 6) in vec4 aWeights;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat4 finalBonesMatrices[100];
        uniform bool useSkinning;

        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoords;

        void main() {
            vec4 totalPosition = vec4(0.0);
            vec3 totalNormal = vec3(0.0);
            
            if (useSkinning) {
                for (int i = 0; i < 4; i++) {
                    if (aBoneIds[i] == -1) continue;
                    if (aBoneIds[i] >= 100) {
                        totalPosition = vec4(aPos, 1.0);
                        break;
                    }
                    vec4 localPosition = finalBonesMatrices[aBoneIds[i]] * vec4(aPos, 1.0);
                    totalPosition += localPosition * aWeights[i];
                    vec3 localNormal = mat3(finalBonesMatrices[aBoneIds[i]]) * aNormal;
                    totalNormal += localNormal * aWeights[i];
                }
            } else {
                totalPosition = vec4(aPos, 1.0);
                totalNormal = aNormal;
            }
            
            FragPos = vec3(model * totalPosition);
            Normal = mat3(transpose(inverse(model))) * totalNormal;
            TexCoords = aTexCoords;
            
            gl_Position = projection * view * model * totalPosition;
        }
    )";

  const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;

        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoords;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;

        void main() {
            // Simple debug: output solid color based on normal
            vec3 norm = normalize(Normal);
            vec3 color = norm * 0.5 + 0.5; // Map normal to RGB
            FragColor = vec4(color, 1.0);
        }
    )";

  shader = std::make_unique<Shader>();
  shader->loadFromSource(vertexShaderSource, fragmentShaderSource);
}

void MoCapViewer::loadModel(const std::string &path) {
  std::lock_guard<std::mutex> lock(dataMutex);
  model = std::make_unique<Model>(path);
  animator->setModel(model.get());

  // Compute bounding box from all meshes
  modelBounds = BoundingBox();
  for (const auto &mesh : model->meshes) {
    for (const auto &vertex : mesh.vertices) {
      modelBounds.min = glm::min(modelBounds.min, vertex.Position);
      modelBounds.max = glm::max(modelBounds.max, vertex.Position);
    }
  }

  std::cout << "Model bounds: (" << modelBounds.min.x << ", "
            << modelBounds.min.y << ", " << modelBounds.min.z << ") to ("
            << modelBounds.max.x << ", " << modelBounds.max.y << ", "
            << modelBounds.max.z << ")" << std::endl;
  std::cout << "Model radius: " << modelBounds.radius() << std::endl;

  // Auto-frame camera to model
  camera->frameModel(modelBounds);
}

void MoCapViewer::run() {
  running = true;

  while (running && !glfwWindowShouldClose(window)) {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    glfwPollEvents();

    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    processInput();
    render();
    renderImGui();

    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  running = false;
}

void MoCapViewer::runAsync() {
  std::thread([this]() { run(); }).detach();
}

void MoCapViewer::stop() { running = false; }

void MoCapViewer::resetCamera() {
  if (modelBounds.isValid()) {
    camera->reset(modelBounds);
  }
}

void MoCapViewer::renderImGui() {
  ImGuiIO &io = ImGui::GetIO();

  // Camera & Controls Panel
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Camera & Controls")) {
    ImGui::Text("FPS: %.1f", io.Framerate);
    ImGui::Separator();

    // Camera Settings
    if (ImGui::CollapsingHeader("Camera Settings",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::SliderFloat("Movement Speed", &camera->MovementSpeed, 0.1f, 20.0f);
      ImGui::SliderFloat("Mouse Sensitivity", &camera->MouseSensitivity, 0.01f,
                         1.0f);
      ImGui::SliderFloat("Scroll Speed", &camera->ScrollSpeed, 0.1f, 10.0f);
      ImGui::SliderFloat("Orbit Speed", &camera->OrbitSpeed, 0.05f, 2.0f);

      if (ImGui::Button("Reset Camera")) {
        resetCamera();
      }
    }

    // Model Info
    if (ImGui::CollapsingHeader("Model Info", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (modelBounds.isValid()) {
        glm::vec3 center = modelBounds.center();
        glm::vec3 size = modelBounds.size();
        ImGui::Text("Center: (%.3f, %.3f, %.3f)", center.x, center.y, center.z);
        ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);
        ImGui::Text("Radius: %.3f", modelBounds.radius());
      } else {
        ImGui::Text("No model loaded");
      }
    }

    // Debug Info
    if (ImGui::CollapsingHeader("Debug Info")) {
      ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera->Position.x,
                  camera->Position.y, camera->Position.z);
      ImGui::Text("Camera Pivot: (%.2f, %.2f, %.2f)", camera->Pivot.x,
                  camera->Pivot.y, camera->Pivot.z);
      ImGui::Text("Orbit Distance: %.2f", camera->OrbitDistance);
      ImGui::Text("Yaw: %.1f, Pitch: %.1f", camera->Yaw, camera->Pitch);
    }

    // Display Settings
    if (ImGui::CollapsingHeader("Display")) {
      ImGui::Checkbox("Show Grid", &showGrid);
      ImGui::ColorEdit3("Background", &backgroundColor.x);
    }
  }
  ImGui::End();

  // Controls Help
  ImGui::SetNextWindowPos(ImVec2(width - 260.0f, 10), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("WASD - Move camera");
    ImGui::Text("Q/E - Move up/down");
    ImGui::Text("Right Mouse - Look around");
    ImGui::Text("Middle Mouse - Orbit");
    ImGui::Text("Scroll - Zoom");
    ImGui::Text("F - Frame model");
    ImGui::Text("ESC - Quit");
  }
  ImGui::End();
}

void MoCapViewer::processInput() {
  // Don't process keyboard input if ImGui wants it
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureKeyboard)
    return;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera->processKeyboard(FORWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera->processKeyboard(BACKWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera->processKeyboard(LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera->processKeyboard(RIGHT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    camera->processKeyboard(UP, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    camera->processKeyboard(DOWN, deltaTime);

  // F key to frame/reset camera
  static bool fKeyWasPressed = false;
  if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
    if (!fKeyWasPressed) {
      resetCamera();
      fKeyWasPressed = true;
    }
  } else {
    fKeyWasPressed = false;
  }
}

void MoCapViewer::render() {
  glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  shader->use();

  // Use camera's dynamic near/far planes based on model size
  glm::mat4 projection =
      glm::perspective(glm::radians(camera->Zoom), (float)width / (float)height,
                       camera->getNearPlane(), camera->getFarPlane());
  glm::mat4 view = camera->getViewMatrix();

  shader->setMat4("projection", projection);
  shader->setMat4("view", view);

  // Position light relative to model
  glm::vec3 lightPos = camera->Pivot + glm::vec3(camera->ModelRadius * 2.0f,
                                                 camera->ModelRadius * 4.0f,
                                                 camera->ModelRadius * 2.0f);
  shader->setVec3("lightPos", lightPos);
  shader->setVec3("viewPos", camera->Position);
  shader->setVec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
  shader->setVec3("objectColor", glm::vec3(0.7f, 0.7f, 0.8f));

  if (showGrid) {
    drawGrid();
  }

  if (model) {
    std::lock_guard<std::mutex> lock(dataMutex);

    glm::mat4 modelMatrix = glm::mat4(1.0f);
    shader->setMat4("model", modelMatrix);
    shader->setBool("useSkinning", false);

    model->draw(*shader);
  }
}

void MoCapViewer::drawGrid() {
  // Simple grid rendering would go here
}

void MoCapViewer::setBoneTransform(const std::string &boneName, float px,
                                   float py, float pz, float qw, float qx,
                                   float qy, float qz, float sx, float sy,
                                   float sz) {
  std::lock_guard<std::mutex> lock(dataMutex);
  BoneTransform transform;
  transform.position = glm::vec3(px, py, pz);
  transform.rotation = glm::quat(qw, qx, qy, qz);
  transform.scale = glm::vec3(sx, sy, sz);
  animator->updateBoneTransform(boneName, transform);
}

void MoCapViewer::setBoneMatrices(const std::vector<float> &matrices) {
  std::lock_guard<std::mutex> lock(dataMutex);
  std::vector<glm::mat4> boneMatrices;

  for (size_t i = 0; i < matrices.size(); i += 16) {
    if (i + 16 <= matrices.size()) {
      glm::mat4 mat;
      for (int j = 0; j < 16; j++) {
        mat[j / 4][j % 4] = matrices[i + j];
      }
      boneMatrices.push_back(mat);
    }
  }

  animator->setFinalBoneMatrices(boneMatrices);
}

void MoCapViewer::setAnimationFrame(
    const std::map<std::string, std::vector<float>> &boneTransforms) {
  std::lock_guard<std::mutex> lock(dataMutex);
  std::map<std::string, BoneTransform> transforms;

  for (const auto &[name, data] : boneTransforms) {
    if (data.size() >= 10) {
      BoneTransform t;
      t.position = glm::vec3(data[0], data[1], data[2]);
      t.rotation = glm::quat(data[3], data[4], data[5], data[6]);
      t.scale = glm::vec3(data[7], data[8], data[9]);
      transforms[name] = t;
    }
  }

  animator->setBoneTransforms(transforms);
}

void MoCapViewer::setCameraPosition(float x, float y, float z) {
  camera->Position = glm::vec3(x, y, z);
}

void MoCapViewer::setCameraTarget(float x, float y, float z) {
  camera->Pivot = glm::vec3(x, y, z);
}

void MoCapViewer::setBackgroundColor(float r, float g, float b) {
  backgroundColor = glm::vec3(r, g, b);
}

void MoCapViewer::framebufferSizeCallback(GLFWwindow *window, int width,
                                          int height) {
  glViewport(0, 0, width, height);
  if (currentViewer) {
    currentViewer->width = width;
    currentViewer->height = height;
  }
}

void MoCapViewer::mouseCallback(GLFWwindow *window, double xposIn,
                                double yposIn) {
  if (!currentViewer)
    return;

  // Don't process mouse input if ImGui wants it
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse)
    return;

  float xpos = static_cast<float>(xposIn);
  float ypos = static_cast<float>(yposIn);

  if (currentViewer->firstMouse) {
    currentViewer->lastX = xpos;
    currentViewer->lastY = ypos;
    currentViewer->firstMouse = false;
  }

  float xoffset = xpos - currentViewer->lastX;
  float yoffset = currentViewer->lastY - ypos;

  currentViewer->lastX = xpos;
  currentViewer->lastY = ypos;

  // Right mouse - orbit around pivot
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
    currentViewer->camera->orbit(xoffset, yoffset, currentViewer->deltaTime);
  }

  // Middle mouse - pan
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
    currentViewer->camera->pan(xoffset, yoffset, currentViewer->deltaTime);
  }
}

void MoCapViewer::scrollCallback(GLFWwindow *window, double xoffset,
                                 double yoffset) {
  if (!currentViewer)
    return;

  // Don't process scroll if ImGui wants it
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse)
    return;

  currentViewer->camera->processMouseScroll(static_cast<float>(yoffset),
                                            currentViewer->deltaTime);
}

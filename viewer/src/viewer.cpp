#include "viewer.h"
#include <cfloat>
#include <cmath>
#include <functional>
#include <iostream>

#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuizmo.h"

static MoCapViewer *currentViewer = nullptr;

MoCapViewer::MoCapViewer(int width, int height, const std::string &title)
    : width(width), height(height), title(title), window(nullptr),
      backgroundColor(0.2f, 0.2f, 0.25f), running(false), lastX(width / 2.0f),
      lastY(height / 2.0f), firstMouse(true), deltaTime(0.0f), lastFrame(0.0f),
      showGrid(true), showImGuiDemo(false), showGizmo(true),
      showSkeleton(false), showSkeletonOnly(false),
      skeletonLineWidth(2.0f), jointSphereSize(0.02f),
      boneColor(0.0f, 0.8f, 1.0f), jointColor(1.0f, 0.4f, 0.0f),
      skeletonVAO(0), skeletonVBO(0), jointVAO(0), jointVBO(0), jointEBO(0), jointIndexCount(0),
      gizmoOperation(ImGuizmo::TRANSLATE), gizmoMode(ImGuizmo::WORLD),
      modelMatrix(1.0f), useSnap(false), snapRotation(15.0f), snapScale(0.5f),
      groundPlaneVAO(0), groundPlaneVBO(0) {
  snapTranslation[0] = 1.0f;
  snapTranslation[1] = 1.0f;
  snapTranslation[2] = 1.0f;
}

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
            vec3 norm = Normal;
            float len = length(norm);
            
            vec3 color;
            if (len > 0.001) {
                // Valid normals - use colorful normal-based shading
                norm = norm / len;
                color = norm * 0.5 + 0.5; // Map normal to RGB (cyan/magenta look)
            } else {
                // Invalid/zero normals - use position-based coloring as fallback
                vec3 posColor = normalize(FragPos) * 0.5 + 0.5;
                color = mix(objectColor, posColor, 0.5);
            }
            
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

  // Load animation from the same FBX file
  modelPath = path;
  animator->loadAnimation(path, 0);

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
    ImGuizmo::BeginFrame();

    processInput();

    // Update animation only if no external pose data is being used
    if (!animator->hasExternalTransforms()) {
      animator->updateAnimation(deltaTime);
    }

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
  // On macOS, GLFW must poll events on main thread
  // Mark as running - caller should use runOneFrame() in their own loop
  running = true;
}

bool MoCapViewer::runOneFrame() {
  if (!running || glfwWindowShouldClose(window)) {
    running = false;
    return false;
  }
  
  float currentFrame = static_cast<float>(glfwGetTime());
  deltaTime = currentFrame - lastFrame;
  lastFrame = currentFrame;

  glfwPollEvents();

  // Start ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGuizmo::BeginFrame();

  processInput();

  // Update animation only if no external pose data is being used
  if (!animator->hasExternalTransforms()) {
    animator->updateAnimation(deltaTime);
  }

  render();
  renderImGui();

  // Render ImGui
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  glfwSwapBuffers(window);
  
  return true;
}

bool MoCapViewer::shouldClose() const {
  return window ? glfwWindowShouldClose(window) : true;
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

    // Animation Controls
    if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
      Animation *anim = animator->getCurrentAnimation();
      if (anim) {
        ImGui::Text("Animation: %s", anim->getName().empty()
                                         ? "Unnamed"
                                         : anim->getName().c_str());
        ImGui::Text("Duration: %.2f sec", animator->getDuration());

        // Play/Pause/Stop buttons
        if (animator->isPlaying()) {
          if (ImGui::Button("Pause"))
            animator->pause();
        } else {
          if (ImGui::Button("Play"))
            animator->play();
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop"))
          animator->stop();

        // Timeline slider
        float progress = animator->getProgress();
        if (ImGui::SliderFloat("Timeline", &progress, 0.0f, 1.0f)) {
          float newTime = progress * anim->getDuration();
          animator->setTime(newTime);
        }

        // Speed control
        float speed = animator->getSpeed();
        if (ImGui::SliderFloat("Speed", &speed, 0.0f, 3.0f)) {
          animator->setSpeed(speed);
        }

        // Loop toggle
        bool looping = animator->isLooping();
        if (ImGui::Checkbox("Loop", &looping)) {
          animator->setLooping(looping);
        }
      } else {
        ImGui::Text("No animation loaded");
      }
    }

    // Transform Gizmo
    if (ImGui::CollapsingHeader("Transform Gizmo", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Checkbox("Show Gizmo", &showGizmo);
      
      if (showGizmo) {
        // Operation selection
        if (ImGui::RadioButton("Translate", gizmoOperation == ImGuizmo::TRANSLATE))
          gizmoOperation = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", gizmoOperation == ImGuizmo::ROTATE))
          gizmoOperation = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", gizmoOperation == ImGuizmo::SCALE))
          gizmoOperation = ImGuizmo::SCALE;
        
        // Mode selection (not for scale)
        if (gizmoOperation != ImGuizmo::SCALE) {
          if (ImGui::RadioButton("Local", gizmoMode == ImGuizmo::LOCAL))
            gizmoMode = ImGuizmo::LOCAL;
          ImGui::SameLine();
          if (ImGui::RadioButton("World", gizmoMode == ImGuizmo::WORLD))
            gizmoMode = ImGuizmo::WORLD;
        }
        
        // Snap settings
        ImGui::Checkbox("Snap", &useSnap);
        if (useSnap) {
          if (gizmoOperation == ImGuizmo::TRANSLATE) {
            ImGui::InputFloat3("Snap XYZ", snapTranslation);
          } else if (gizmoOperation == ImGuizmo::ROTATE) {
            ImGui::InputFloat("Angle Snap", &snapRotation);
          } else if (gizmoOperation == ImGuizmo::SCALE) {
            ImGui::InputFloat("Scale Snap", &snapScale);
          }
        }
        
        // Transform display
        float translation[3], rotation[3], scale[3];
        ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(modelMatrix), 
                                               translation, rotation, scale);
        bool changed = false;
        changed |= ImGui::InputFloat3("Position", translation);
        changed |= ImGui::InputFloat3("Rotation", rotation);
        changed |= ImGui::InputFloat3("Scale", scale);
        if (changed) {
          ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale,
                                                   glm::value_ptr(modelMatrix));
        }
        
        if (ImGui::Button("Reset Transform")) {
          modelMatrix = glm::mat4(1.0f);
        }
      }
    }
    
    // Display Settings
    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Checkbox("Show Grid", &showGrid);
      ImGui::Separator();
      
      // Skeleton visualization (like Unreal Engine)
      ImGui::Text("Skeleton View");
      ImGui::Checkbox("Show Skeleton", &showSkeleton);
      ImGui::Checkbox("Skeleton Only", &showSkeletonOnly);
      
      if (showSkeleton || showSkeletonOnly) {
        // Note: Bone Width slider removed - glLineWidth > 1.0 not supported on macOS
        ImGui::SliderFloat("Joint Size", &jointSphereSize, 0.005f, 0.1f);
        ImGui::ColorEdit3("Bone Color", &boneColor.x);
        ImGui::ColorEdit3("Joint Color", &jointColor.x);
      }
      
      ImGui::Separator();
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

  // Skeleton Bones Hierarchy Window
  ImGui::SetNextWindowPos(ImVec2(width - 260.0f, 200), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(250, 400), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Skeleton Bones")) {
    if (model && model->getHasSkeleton()) {
      ImGui::Text("Bones: %d", model->getBoneCount());
      ImGui::Separator();
      renderBoneHierarchy(model->getRootBone());
    } else if (model) {
      ImGui::Text("Static Model (No Skeleton)");
      ImGui::Text("Meshes: %zu", model->meshes.size());
    } else {
      ImGui::Text("No model loaded");
    }
  }
  ImGui::End();
}

void MoCapViewer::renderBoneHierarchy(const BoneNode &node) {
  // Determine if this is a bone (has valid bone ID) or just a scene node
  bool isBone = node.boneId >= 0;

  // Create tree node flags
  ImGuiTreeNodeFlags flags =
      ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
  if (node.children.empty()) {
    flags |= ImGuiTreeNodeFlags_Leaf;
  }

  // Format the node label
  std::string label;
  if (isBone) {
    label = node.name + " [" + std::to_string(node.boneId) + "]";
  } else {
    label = node.name;
  }

  // Set color for bones vs regular nodes
  if (isBone) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
  }

  bool isOpen = ImGui::TreeNodeEx(label.c_str(), flags);

  if (isBone) {
    ImGui::PopStyleColor();
  }

  if (isOpen) {
    for (const auto &child : node.children) {
      renderBoneHierarchy(child);
    }
    ImGui::TreePop();
  }
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

    shader->setMat4("model", modelMatrix);

    // Enable skinning if we have animation OR external bone transforms
    Animation *anim = animator->getCurrentAnimation();
    bool hasAnimation = anim && anim->isValid();
    bool hasExternalPose = animator->hasExternalTransforms();
    bool useSkinning = hasAnimation || hasExternalPose;
    
    static bool debugOnce = true;
    if (debugOnce && hasExternalPose) {
      std::cout << "SKINNING ENABLED: hasAnim=" << hasAnimation 
                << " hasExternal=" << hasExternalPose << std::endl;
      debugOnce = false;
    }
    
    shader->setBool("useSkinning", useSkinning);

    if (useSkinning) {
      auto &boneMatrices = animator->getFinalBoneMatrices();
      for (int i = 0; i < std::min(100, (int)boneMatrices.size()); i++) {
        shader->setMat4("finalBonesMatrices[" + std::to_string(i) + "]",
                        boneMatrices[i]);
      }
    }

    // Draw mesh (unless skeleton-only mode)
    if (!showSkeletonOnly) {
      model->draw(*shader);
    }
    
    // Draw skeleton overlay
    if (showSkeleton || showSkeletonOnly) {
      renderSkeleton();
    }
  }
  
  // Render ImGuizmo gizmo
  if (showGizmo && model) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    
    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 proj = glm::perspective(glm::radians(camera->Zoom), 
                                       (float)width / (float)height,
                                       camera->getNearPlane(), 
                                       camera->getFarPlane());
    
    float* snapPtr = nullptr;
    if (useSnap) {
      if (gizmoOperation == ImGuizmo::TRANSLATE) {
        snapPtr = snapTranslation;
      } else if (gizmoOperation == ImGuizmo::ROTATE) {
        snapPtr = &snapRotation;
      } else if (gizmoOperation == ImGuizmo::SCALE) {
        snapPtr = &snapScale;
      }
    }
    
    ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                         static_cast<ImGuizmo::OPERATION>(gizmoOperation),
                         static_cast<ImGuizmo::MODE>(gizmoMode),
                         glm::value_ptr(modelMatrix),
                         nullptr, snapPtr);
  }
}

void MoCapViewer::drawGrid() {
  // Generate grid vertices on first call
  static unsigned int gridVAO = 0, gridVBO = 0;
  static int gridVertexCount = 0;
  
  float gridSize = camera->ModelRadius * 2.0f;
  int gridLines = 20;
  float step = gridSize / gridLines;
  float halfSize = gridSize / 2.0f;
  
  if (gridVAO == 0) {
    std::vector<float> vertices;
    
    // Generate grid lines along X axis
    for (int i = -gridLines; i <= gridLines; i++) {
      float pos = i * step;
      // Line along Z
      vertices.push_back(-halfSize); vertices.push_back(0.0f); vertices.push_back(pos);
      vertices.push_back(halfSize);  vertices.push_back(0.0f); vertices.push_back(pos);
    }
    
    // Generate grid lines along Z axis
    for (int i = -gridLines; i <= gridLines; i++) {
      float pos = i * step;
      // Line along X
      vertices.push_back(pos); vertices.push_back(0.0f); vertices.push_back(-halfSize);
      vertices.push_back(pos); vertices.push_back(0.0f); vertices.push_back(halfSize);
    }
    
    gridVertexCount = vertices.size() / 3;
    
    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);
    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
  }
  
  // Draw grid with depth test but write behind other objects
  shader->setMat4("model", glm::mat4(1.0f));
  shader->setBool("useSkinning", false);
  shader->setVec3("objectColor", glm::vec3(0.4f, 0.4f, 0.45f));
  
  glBindVertexArray(gridVAO);
  glDrawArrays(GL_LINES, 0, gridVertexCount);
  glBindVertexArray(0);
}

void MoCapViewer::setupSkeletonBuffers() {
  // Create VAO/VBO for skeleton lines (will be updated dynamically)
  if (skeletonVAO == 0) {
    glGenVertexArrays(1, &skeletonVAO);
    glGenBuffers(1, &skeletonVBO);
  }
  
  // Create sphere geometry for joints
  if (jointVAO == 0) {
    const int segments = 16;
    const int rings = 12;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    // Generate sphere vertices
    for (int r = 0; r <= rings; r++) {
      float phi = M_PI * r / rings;
      for (int s = 0; s <= segments; s++) {
        float theta = 2.0f * M_PI * s / segments;
        float x = sin(phi) * cos(theta);
        float y = cos(phi);
        float z = sin(phi) * sin(theta);
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
      }
    }
    
    // Generate indices for triangle strip
    for (int r = 0; r < rings; r++) {
      for (int s = 0; s < segments; s++) {
        int curr = r * (segments + 1) + s;
        int next = curr + segments + 1;
        indices.push_back(curr);
        indices.push_back(next);
        indices.push_back(curr + 1);
        indices.push_back(curr + 1);
        indices.push_back(next);
        indices.push_back(next + 1);
      }
    }
    
    jointIndexCount = indices.size();
    
    glGenVertexArrays(1, &jointVAO);
    glGenBuffers(1, &jointVBO);
    glGenBuffers(1, &jointEBO);
    
    glBindVertexArray(jointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, jointVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, jointEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
  }
}

void MoCapViewer::collectBonePositions(const BoneNode& node, const glm::mat4& parentTransform,
                                       std::vector<glm::vec3>& boneLines,
                                       std::vector<glm::vec3>& jointPositions) {
  // Calculate this bone's world position
  glm::mat4 nodeTransform = parentTransform * node.transform;
  glm::vec3 nodePos = glm::vec3(nodeTransform[3]);
  
  // Add joint position
  jointPositions.push_back(nodePos);
  
  // Draw lines to children
  for (const auto& child : node.children) {
    glm::mat4 childTransform = nodeTransform * child.transform;
    glm::vec3 childPos = glm::vec3(childTransform[3]);
    
    // Add line from parent to child
    boneLines.push_back(nodePos);
    boneLines.push_back(childPos);
    
    // Recurse
    collectBonePositions(child, nodeTransform, boneLines, jointPositions);
  }
}

void MoCapViewer::renderSkeleton() {
  if (!model || !model->getHasSkeleton()) return;
  
  // Clear any pending GL errors before skeleton rendering
  while (glGetError() != GL_NO_ERROR) {}
  
  setupSkeletonBuffers();
  
  // Collect bone positions from hierarchy
  std::vector<glm::vec3> boneLines;
  std::vector<glm::vec3> jointPositions;
  
  const BoneNode& root = model->getRootBone();
  glm::mat4 rootTransform = modelMatrix;
  
  // If we have animation, use the animated bone positions
  auto& boneMatrices = animator->getFinalBoneMatrices();
  auto& boneInfoMap = model->getBoneInfoMap();
  
  if (!boneMatrices.empty() && animator->hasExternalTransforms()) {
    // Use animated positions - traverse bone hierarchy and apply transforms
    std::function<void(const BoneNode&, const glm::mat4&)> collectAnimated;
    collectAnimated = [&](const BoneNode& node, const glm::mat4& parentWorld) {
      glm::mat4 nodeWorld = parentWorld;
      
      // Check if this is a bone with animation
      auto it = boneInfoMap.find(node.name);
      if (it != boneInfoMap.end() && it->second.id < (int)boneMatrices.size()) {
        // Apply the animated bone matrix
        glm::mat4 animatedLocal = boneMatrices[it->second.id] * glm::inverse(it->second.offset);
        nodeWorld = modelMatrix * animatedLocal;
      } else {
        nodeWorld = parentWorld * node.transform;
      }
      
      glm::vec3 nodePos = glm::vec3(nodeWorld[3]);
      jointPositions.push_back(nodePos);
      
      for (const auto& child : node.children) {
        glm::mat4 childWorld = nodeWorld;
        auto childIt = boneInfoMap.find(child.name);
        if (childIt != boneInfoMap.end() && childIt->second.id < (int)boneMatrices.size()) {
          glm::mat4 childAnimated = boneMatrices[childIt->second.id] * glm::inverse(childIt->second.offset);
          childWorld = modelMatrix * childAnimated;
        } else {
          childWorld = nodeWorld * child.transform;
        }
        
        glm::vec3 childPos = glm::vec3(childWorld[3]);
        boneLines.push_back(nodePos);
        boneLines.push_back(childPos);
        
        collectAnimated(child, nodeWorld);
      }
    };
    
    collectAnimated(root, rootTransform);
  } else {
    // Use bind pose
    collectBonePositions(root, rootTransform, boneLines, jointPositions);
  }
  
  // Scale joint size based on model
  float jointSize = jointSphereSize * camera->ModelRadius;
  
  // Render bone lines
  if (!boneLines.empty()) {
    glBindVertexArray(skeletonVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skeletonVBO);
    glBufferData(GL_ARRAY_BUFFER, boneLines.size() * sizeof(glm::vec3), boneLines.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    
    shader->setMat4("model", glm::mat4(1.0f));
    shader->setBool("useSkinning", false);
    shader->setVec3("objectColor", boneColor);
    
    // Note: glLineWidth > 1.0 not supported on macOS Core Profile
    // Line width setting removed to avoid GL_INVALID_VALUE errors
    glDrawArrays(GL_LINES, 0, boneLines.size());
    glBindVertexArray(0);
  }
  
  // Render joint spheres
  if (!jointPositions.empty()) {
    shader->setVec3("objectColor", jointColor);
    
    for (const auto& pos : jointPositions) {
      glm::mat4 jointModel = glm::translate(glm::mat4(1.0f), pos);
      jointModel = glm::scale(jointModel, glm::vec3(jointSize));
      shader->setMat4("model", jointModel);
      
      glBindVertexArray(jointVAO);
      glDrawElements(GL_TRIANGLES, jointIndexCount, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
    }
  }
  
  // Clear any GL errors from skeleton rendering to prevent spam in mesh draw
  while (glGetError() != GL_NO_ERROR) {}
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
  
  static bool firstCall = true;
  if (firstCall && model) {
    auto &boneInfoMap = model->getBoneInfoMap();
    std::cout << "Available bones in model (" << boneInfoMap.size() << "):" << std::endl;
    for (const auto &[name, info] : boneInfoMap) {
      std::cout << "  [" << info.id << "] " << name << std::endl;
    }
    firstCall = false;
  }

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

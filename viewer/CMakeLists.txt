cmake_minimum_required(VERSION 3.16)
project(MoCapViewer VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)
find_package(pybind11 REQUIRED)

# ImGui source files
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vendor/imgui-1.92.5)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ImGuizmo source files
set(IMGUIZMO_DIR ${CMAKE_SOURCE_DIR}/vendor/ImGuizmo-1.83)
set(IMGUIZMO_SOURCES
    ${IMGUIZMO_DIR}/ImGuizmo.cpp
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor/glad/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMGUIZMO_DIR}
    /opt/homebrew/include
)

# Link directories
link_directories(/opt/homebrew/lib)

# Source files
set(VIEWER_SOURCES
    src/main.cpp
    src/viewer.cpp
    src/shader.cpp
    src/mesh.cpp
    src/model.cpp
    src/animator.cpp
    src/animation.cpp
    src/camera.cpp
    vendor/glad/src/gl.c
    ${IMGUI_SOURCES}
    ${IMGUIZMO_SOURCES}
)

# Standalone executable
add_executable(mocap_viewer ${VIEWER_SOURCES})
target_link_libraries(mocap_viewer
    OpenGL::GL
    glfw
    assimp
)

# Python module
pybind11_add_module(mocap_viewer_py
    src/viewer.cpp
    src/shader.cpp
    src/mesh.cpp
    src/model.cpp
    src/animator.cpp
    src/animation.cpp
    src/camera.cpp
    vendor/glad/src/gl.c
    ${IMGUI_SOURCES}
    ${IMGUIZMO_SOURCES}
    src/python_bindings.cpp
)
target_link_libraries(mocap_viewer_py PRIVATE
    OpenGL::GL
    glfw
    assimp
)
target_compile_definitions(mocap_viewer_py PRIVATE PYTHON_MODULE)

# Copy shaders to build directory
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})

# Install targets
install(TARGETS mocap_viewer DESTINATION bin)
install(TARGETS mocap_viewer_py DESTINATION lib)

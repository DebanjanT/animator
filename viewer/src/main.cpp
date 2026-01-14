#ifndef PYTHON_MODULE

#include "viewer.h"
#include <iostream>

int main(int argc, char *argv[]) {
  std::string modelPath = "";

  if (argc > 1) {
    modelPath = argv[1];
  }

  MoCapViewer viewer(1280, 720, "MoCap to UE5 - 3D Viewer");

  if (!viewer.initialize()) {
    std::cerr << "Failed to initialize viewer" << std::endl;
    return -1;
  }

  if (!modelPath.empty()) {
    std::cout << "Loading model: " << modelPath << std::endl;
    viewer.loadModel(modelPath);
  }

  std::cout << "Controls:" << std::endl;
  std::cout << "  WASD - Move camera" << std::endl;
  std::cout << "  QE - Move up/down" << std::endl;
  std::cout << "  Right Mouse - Look around" << std::endl;
  std::cout << "  Middle Mouse - Orbit" << std::endl;
  std::cout << "  Scroll - Zoom" << std::endl;
  std::cout << "  ESC - Quit" << std::endl;

  viewer.run();

  return 0;
}

#endif

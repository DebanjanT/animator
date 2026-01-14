#include "viewer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(mocap_viewer_py, m) {
  m.doc() = "MoCap 3D Viewer with OpenGL - Python bindings";

  py::class_<MoCapViewer>(m, "MoCapViewer")
      .def(py::init<int, int, const std::string &>(), py::arg("width") = 1280,
           py::arg("height") = 720, py::arg("title") = "MoCap Viewer")
      .def("initialize", &MoCapViewer::initialize,
           "Initialize the OpenGL context and window")
      .def("load_model", &MoCapViewer::loadModel, py::arg("path"),
           "Load a 3D model (FBX, OBJ, etc.)")
      .def("run", &MoCapViewer::run, "Run the viewer main loop (blocking)")
      .def("run_async", &MoCapViewer::runAsync,
           "Mark viewer as running for frame-by-frame processing")
      .def("run_one_frame", &MoCapViewer::runOneFrame,
           "Process single frame, returns False if should quit")
      .def("should_close", &MoCapViewer::shouldClose,
           "Check if window should close")
      .def("stop", &MoCapViewer::stop, "Stop the viewer")
      .def("is_running", &MoCapViewer::isRunning,
           "Check if the viewer is running")
      .def("set_bone_transform", &MoCapViewer::setBoneTransform,
           py::arg("bone_name"), py::arg("px"), py::arg("py"), py::arg("pz"),
           py::arg("qw"), py::arg("qx"), py::arg("qy"), py::arg("qz"),
           py::arg("sx"), py::arg("sy"), py::arg("sz"),
           "Set transform for a single bone")
      .def("set_bone_matrices", &MoCapViewer::setBoneMatrices,
           py::arg("matrices"),
           "Set all bone matrices as a flat array of floats (16 per matrix)")
      .def("set_animation_frame", &MoCapViewer::setAnimationFrame,
           py::arg("bone_transforms"),
           "Set animation frame from dict of bone name -> "
           "[px,py,pz,qw,qx,qy,qz,sx,sy,sz]")
      .def("set_camera_position", &MoCapViewer::setCameraPosition, py::arg("x"),
           py::arg("y"), py::arg("z"), "Set camera position")
      .def("set_camera_target", &MoCapViewer::setCameraTarget, py::arg("x"),
           py::arg("y"), py::arg("z"), "Set camera look-at target")
      .def("set_background_color", &MoCapViewer::setBackgroundColor,
           py::arg("r"), py::arg("g"), py::arg("b"),
           "Set background color (RGB 0-1)")
      .def_property_readonly("width", &MoCapViewer::getWidth)
      .def_property_readonly("height", &MoCapViewer::getHeight);
}

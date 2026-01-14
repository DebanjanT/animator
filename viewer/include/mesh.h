#ifndef MESH_H
#define MESH_H

#include "shader.h"
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#define MAX_BONE_INFLUENCE 4

struct Vertex {
  glm::vec3 Position;
  glm::vec3 Normal;
  glm::vec2 TexCoords;
  glm::vec3 Tangent;
  glm::vec3 Bitangent;
  int BoneIDs[MAX_BONE_INFLUENCE];
  float Weights[MAX_BONE_INFLUENCE];
};

struct Texture {
  unsigned int id;
  std::string type;
  std::string path;
};

struct BoneInfo {
  int id;
  glm::mat4 offset;
};

class Mesh {
public:
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  std::vector<Texture> textures;
  unsigned int VAO;

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices,
       std::vector<Texture> textures);
  void draw(Shader &shader);

private:
  unsigned int VBO, EBO;
  void setupMesh();
};

#endif

#ifndef MODEL_H
#define MODEL_H

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <map>
#include <string>
#include <vector>

#include "mesh.h"
#include "shader.h"

class Model {
public:
  std::vector<Texture> textures_loaded;
  std::vector<Mesh> meshes;
  std::string directory;
  std::map<std::string, BoneInfo> boneInfoMap;
  int boneCounter = 0;

  Model() = default;
  Model(const std::string &path);

  void load(const std::string &path);
  void draw(Shader &shader);

  auto &getBoneInfoMap() { return boneInfoMap; }
  int &getBoneCount() { return boneCounter; }

private:
  void processNode(aiNode *node, const aiScene *scene);
  Mesh processMesh(aiMesh *mesh, const aiScene *scene);
  std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type,
                                            const std::string &typeName);

  void extractBoneWeightForVertices(std::vector<Vertex> &vertices, aiMesh *mesh,
                                    const aiScene *scene);
  void setVertexBoneDataToDefault(Vertex &vertex);
  void setVertexBoneData(Vertex &vertex, int boneID, float weight);
};

#endif

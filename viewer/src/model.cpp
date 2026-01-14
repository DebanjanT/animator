#include "model.h"
#include <cfloat>
#include <iostream>

Model::Model(const std::string &path) { load(path); }

void Model::load(const std::string &path) {
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(
      path, aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                aiProcess_FlipUVs | aiProcess_CalcTangentSpace |
                aiProcess_LimitBoneWeights);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
    return;
  }

  directory = path.substr(0, path.find_last_of('/'));
  processNode(scene->mRootNode, scene);

  // Build bone hierarchy for display
  if (boneCounter > 0) {
    hasSkeleton = true;
    buildBoneHierarchy(scene->mRootNode, rootBone);
  }

  std::cout << "Model loaded: " << path << std::endl;
  std::cout << "  Meshes: " << meshes.size() << std::endl;
  std::cout << "  Bones: " << boneCounter << std::endl;
  
  // Log all bone names for analysis
  if (boneCounter > 0) {
    std::cout << "\n=== BONE HIERARCHY ===" << std::endl;
    logBoneHierarchy(rootBone, 0);
    std::cout << "=== END BONE HIERARCHY ===\n" << std::endl;
  }
}

void Model::draw(Shader &shader) {
  for (auto &mesh : meshes) {
    mesh.draw(shader);
  }
}

void Model::processNode(aiNode *node, const aiScene *scene) {
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    meshes.push_back(processMesh(mesh, scene));
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    processNode(node->mChildren[i], scene);
  }
}

Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene) {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  std::vector<Texture> textures;

  glm::vec3 minPos(FLT_MAX), maxPos(-FLT_MAX);
  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    Vertex vertex;
    setVertexBoneDataToDefault(vertex);

    vertex.Position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y,
                                mesh->mVertices[i].z);
    minPos = glm::min(minPos, vertex.Position);
    maxPos = glm::max(maxPos, vertex.Position);

    if (mesh->HasNormals()) {
      vertex.Normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y,
                                mesh->mNormals[i].z);
    } else {
      vertex.Normal = glm::vec3(0.0f, 1.0f, 0.0f); // Default up normal
    }

    if (mesh->mTextureCoords[0]) {
      vertex.TexCoords =
          glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);

      if (mesh->mTangents) {
        vertex.Tangent = glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y,
                                   mesh->mTangents[i].z);
      }

      if (mesh->mBitangents) {
        vertex.Bitangent =
            glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y,
                      mesh->mBitangents[i].z);
      }
    } else {
      vertex.TexCoords = glm::vec2(0.0f, 0.0f);
    }

    vertices.push_back(vertex);
  }

  std::cout << "  Bounding box: (" << minPos.x << ", " << minPos.y << ", "
            << minPos.z << ") to (" << maxPos.x << ", " << maxPos.y << ", "
            << maxPos.z << ")" << std::endl;
  std::cout << "  Has normals: " << (mesh->HasNormals() ? "yes" : "no")
            << std::endl;

  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (unsigned int j = 0; j < face.mNumIndices; j++) {
      indices.push_back(face.mIndices[j]);
    }
  }

  extractBoneWeightForVertices(vertices, mesh, scene);

  return Mesh(vertices, indices, textures);
}

void Model::setVertexBoneDataToDefault(Vertex &vertex) {
  for (int i = 0; i < MAX_BONE_INFLUENCE; i++) {
    vertex.BoneIDs[i] = -1;
    vertex.Weights[i] = 0.0f;
  }
}

void Model::setVertexBoneData(Vertex &vertex, int boneID, float weight) {
  for (int i = 0; i < MAX_BONE_INFLUENCE; i++) {
    if (vertex.BoneIDs[i] < 0) {
      vertex.Weights[i] = weight;
      vertex.BoneIDs[i] = boneID;
      break;
    }
  }
}

void Model::extractBoneWeightForVertices(std::vector<Vertex> &vertices,
                                         aiMesh *mesh, const aiScene *scene) {
  for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; boneIndex++) {
    int boneID = -1;
    std::string boneName = mesh->mBones[boneIndex]->mName.C_Str();

    if (boneInfoMap.find(boneName) == boneInfoMap.end()) {
      BoneInfo newBoneInfo;
      newBoneInfo.id = boneCounter;

      auto &m = mesh->mBones[boneIndex]->mOffsetMatrix;
      newBoneInfo.offset =
          glm::mat4(m.a1, m.b1, m.c1, m.d1, m.a2, m.b2, m.c2, m.d2, m.a3, m.b3,
                    m.c3, m.d3, m.a4, m.b4, m.c4, m.d4);

      boneInfoMap[boneName] = newBoneInfo;
      boneID = boneCounter;
      boneCounter++;
    } else {
      boneID = boneInfoMap[boneName].id;
    }

    auto weights = mesh->mBones[boneIndex]->mWeights;
    int numWeights = mesh->mBones[boneIndex]->mNumWeights;

    for (int weightIndex = 0; weightIndex < numWeights; weightIndex++) {
      int vertexId = weights[weightIndex].mVertexId;
      float weight = weights[weightIndex].mWeight;
      setVertexBoneData(vertices[vertexId], boneID, weight);
    }
  }
}

std::vector<Texture> Model::loadMaterialTextures(aiMaterial *mat,
                                                 aiTextureType type,
                                                 const std::string &typeName) {
  std::vector<Texture> textures;
  // Texture loading would go here - simplified for now
  return textures;
}

static glm::mat4 convertMatrix(const aiMatrix4x4 &from) {
  return glm::mat4(from.a1, from.b1, from.c1, from.d1, from.a2, from.b2,
                   from.c2, from.d2, from.a3, from.b3, from.c3, from.d3,
                   from.a4, from.b4, from.c4, from.d4);
}

void Model::buildBoneHierarchy(const aiNode *node, BoneNode &boneNode) {
  boneNode.name = node->mName.C_Str();
  boneNode.transform = convertMatrix(node->mTransformation);

  // Check if this node is a bone
  auto it = boneInfoMap.find(boneNode.name);
  if (it != boneInfoMap.end()) {
    boneNode.boneId = it->second.id;
  }

  // Recursively process children
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    BoneNode child;
    buildBoneHierarchy(node->mChildren[i], child);
    boneNode.children.push_back(child);
  }
}

void Model::logBoneHierarchy(const BoneNode &node, int depth) {
  std::string indent(depth * 2, ' ');
  std::string boneMarker = (node.boneId >= 0) ? " [BONE:" + std::to_string(node.boneId) + "]" : "";
  std::cout << indent << node.name << boneMarker << std::endl;
  
  for (const auto &child : node.children) {
    logBoneHierarchy(child, depth + 1);
  }
}

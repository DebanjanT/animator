#ifndef ANIMATOR_H
#define ANIMATOR_H

#include "model.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <string>
#include <vector>

struct BoneTransform {
  glm::vec3 position;
  glm::quat rotation;
  glm::vec3 scale;

  BoneTransform()
      : position(0.0f), rotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)),
        scale(1.0f) {}
};

class Animator {
public:
  Animator();
  Animator(Model *model);

  void setModel(Model *model);
  void updateBoneTransform(const std::string &boneName,
                           const BoneTransform &transform);
  void updateBoneTransformByIndex(int boneIndex, const glm::mat4 &transform);
  void
  setBoneTransforms(const std::map<std::string, BoneTransform> &transforms);
  void setFinalBoneMatrices(const std::vector<glm::mat4> &matrices);

  void calculateBoneTransforms();
  std::vector<glm::mat4> &getFinalBoneMatrices() { return finalBoneMatrices; }

  void reset();

  // UE5 Mannequin bone names mapping
  static const std::map<std::string, std::string> UE5_BONE_MAPPING;

private:
  Model *model;
  std::vector<glm::mat4> finalBoneMatrices;
  std::map<std::string, BoneTransform> boneTransforms;

  void calculateBoneTransform(const std::string &boneName,
                              const glm::mat4 &parentTransform);
  glm::mat4 getBoneLocalTransform(const std::string &boneName);
};

#endif

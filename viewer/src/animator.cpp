#define GLM_ENABLE_EXPERIMENTAL
#include "animator.h"
#include <glm/gtx/quaternion.hpp>

const std::map<std::string, std::string> Animator::UE5_BONE_MAPPING = {
    {"pelvis", "pelvis"},         {"spine_01", "spine_01"},
    {"spine_02", "spine_02"},     {"spine_03", "spine_03"},
    {"clavicle_l", "clavicle_l"}, {"upperarm_l", "upperarm_l"},
    {"lowerarm_l", "lowerarm_l"}, {"hand_l", "hand_l"},
    {"clavicle_r", "clavicle_r"}, {"upperarm_r", "upperarm_r"},
    {"lowerarm_r", "lowerarm_r"}, {"hand_r", "hand_r"},
    {"neck_01", "neck_01"},       {"head", "head"},
    {"thigh_l", "thigh_l"},       {"calf_l", "calf_l"},
    {"foot_l", "foot_l"},         {"ball_l", "ball_l"},
    {"thigh_r", "thigh_r"},       {"calf_r", "calf_r"},
    {"foot_r", "foot_r"},         {"ball_r", "ball_r"}};

Animator::Animator() : model(nullptr) {
  finalBoneMatrices.resize(100, glm::mat4(1.0f));
}

Animator::Animator(Model *model) : model(model) {
  finalBoneMatrices.resize(100, glm::mat4(1.0f));
}

void Animator::setModel(Model *model) {
  this->model = model;
  reset();
}

void Animator::reset() {
  for (auto &mat : finalBoneMatrices) {
    mat = glm::mat4(1.0f);
  }
  boneTransforms.clear();
}

void Animator::updateBoneTransform(const std::string &boneName,
                                   const BoneTransform &transform) {
  boneTransforms[boneName] = transform;
}

void Animator::updateBoneTransformByIndex(int boneIndex,
                                          const glm::mat4 &transform) {
  if (boneIndex >= 0 && boneIndex < (int)finalBoneMatrices.size()) {
    finalBoneMatrices[boneIndex] = transform;
  }
}

void Animator::setBoneTransforms(
    const std::map<std::string, BoneTransform> &transforms) {
  boneTransforms = transforms;
  calculateBoneTransforms();
}

void Animator::setFinalBoneMatrices(const std::vector<glm::mat4> &matrices) {
  finalBoneMatrices = matrices;
}

void Animator::calculateBoneTransforms() {
  if (!model)
    return;

  auto &boneInfoMap = model->getBoneInfoMap();

  for (auto &[boneName, transform] : boneTransforms) {
    auto it = boneInfoMap.find(boneName);
    if (it != boneInfoMap.end()) {
      int boneIndex = it->second.id;
      glm::mat4 localTransform = getBoneLocalTransform(boneName);
      finalBoneMatrices[boneIndex] = localTransform * it->second.offset;
    }
  }
}

glm::mat4 Animator::getBoneLocalTransform(const std::string &boneName) {
  auto it = boneTransforms.find(boneName);
  if (it == boneTransforms.end()) {
    return glm::mat4(1.0f);
  }

  const BoneTransform &transform = it->second;

  glm::mat4 translation = glm::translate(glm::mat4(1.0f), transform.position);
  glm::mat4 rotation = glm::toMat4(transform.rotation);
  glm::mat4 scale = glm::scale(glm::mat4(1.0f), transform.scale);

  return translation * rotation * scale;
}

void Animator::calculateBoneTransform(const std::string &boneName,
                                      const glm::mat4 &parentTransform) {
  glm::mat4 localTransform = getBoneLocalTransform(boneName);
  glm::mat4 globalTransform = parentTransform * localTransform;

  if (model) {
    auto &boneInfoMap = model->getBoneInfoMap();
    auto it = boneInfoMap.find(boneName);
    if (it != boneInfoMap.end()) {
      int boneIndex = it->second.id;
      finalBoneMatrices[boneIndex] = globalTransform * it->second.offset;
    }
  }
}

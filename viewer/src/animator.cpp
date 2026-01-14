#define GLM_ENABLE_EXPERIMENTAL
#include "animator.h"
#include <glm/gtx/quaternion.hpp>
#include <iostream>

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

Animator::Animator() { finalBoneMatrices.resize(100, glm::mat4(1.0f)); }

Animator::Animator(Model *model) : model(model) {
  finalBoneMatrices.resize(100, glm::mat4(1.0f));
}

void Animator::setModel(Model *model) {
  this->model = model;
  reset();
}

void Animator::loadAnimation(const std::string &path, int animIndex) {
  ownedAnimation = std::make_unique<Animation>(path, animIndex);
  currentAnimation = ownedAnimation.get();
  animationNames = Animation::getAnimationNames(path);
  currentTime = 0.0f;

  // Initialize bone matrices
  if (currentAnimation) {
    updateAnimation(0.0f);
  }
}

void Animator::playAnimation(Animation *animation) {
  currentAnimation = animation;
  currentTime = 0.0f;
}

void Animator::updateAnimation(float deltaTime) {
  if (!currentAnimation)
    return;

  if (playing) {
    currentTime +=
        currentAnimation->getTicksPerSecond() * deltaTime * playbackSpeed;

    if (looping) {
      currentTime = fmod(currentTime, currentAnimation->getDuration());
      if (currentTime < 0)
        currentTime += currentAnimation->getDuration();
    } else {
      if (currentTime >= currentAnimation->getDuration()) {
        currentTime = currentAnimation->getDuration();
        playing = false;
      }
      if (currentTime < 0)
        currentTime = 0;
    }
  }

  calculateBoneTransform(currentAnimation->getRootNode(), glm::mat4(1.0f));
}

float Animator::getDuration() const {
  if (!currentAnimation)
    return 0.0f;
  return currentAnimation->getDuration() /
         currentAnimation->getTicksPerSecond();
}

float Animator::getProgress() const {
  if (!currentAnimation || currentAnimation->getDuration() == 0)
    return 0.0f;
  return currentTime / currentAnimation->getDuration();
}

void Animator::calculateBoneTransform(const AssimpNodeData &node,
                                      const glm::mat4 &parentTransform) {
  const std::string &nodeName = node.name;
  glm::mat4 nodeTransform = node.transformation;

  Bone *bone = currentAnimation->findBone(nodeName);
  if (bone) {
    bone->update(currentTime);
    nodeTransform = bone->getLocalTransform();
  }

  glm::mat4 globalTransformation = parentTransform * nodeTransform;

  auto &boneInfoMap = currentAnimation->getBoneIDMap();
  auto it = boneInfoMap.find(nodeName);
  if (it != boneInfoMap.end()) {
    int index = it->second.id;
    if (index >= 0 && index < static_cast<int>(finalBoneMatrices.size())) {
      finalBoneMatrices[index] = globalTransformation * it->second.offset;
    }
  }

  for (const auto &child : node.children) {
    calculateBoneTransform(child, globalTransformation);
  }
}

void Animator::reset() {
  for (auto &mat : finalBoneMatrices) {
    mat = glm::mat4(1.0f);
  }
  boneTransforms.clear();
  currentTime = 0.0f;
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
  // Don't reset - just update the bones we have transforms for
  // Other bones keep their current state (identity = bind pose)
  boneTransforms = transforms;
  calculateBoneTransforms();
}

void Animator::setFinalBoneMatrices(const std::vector<glm::mat4> &matrices) {
  finalBoneMatrices = matrices;
}

void Animator::calculateBoneTransforms() {
  if (!model)
    return;

  // Use hierarchical calculation - traverse bone tree from root
  // This ensures child bones inherit parent transforms
  if (model->getHasSkeleton()) {
    const BoneNode &root = model->getRootBone();
    calculateHierarchicalTransforms(root, glm::mat4(1.0f));
  }
  
  static bool firstDebug = true;
  if (firstDebug && !boneTransforms.empty()) {
    auto &boneInfoMap = model->getBoneInfoMap();
    int matchedBones = 0;
    for (auto &[boneName, transform] : boneTransforms) {
      auto it = boneInfoMap.find(boneName);
      if (it != boneInfoMap.end()) {
        glm::quat q = transform.rotation;
        std::cout << "Bone " << boneName << " rotation: w=" << q.w 
                  << " x=" << q.x << " y=" << q.y << " z=" << q.z << std::endl;
        matchedBones++;
      } else {
        std::cout << "Bone NOT found in model: " << boneName << std::endl;
      }
    }
    std::cout << "Matched " << matchedBones << "/" << boneTransforms.size() << " bones" << std::endl;
    firstDebug = false;
  }
}

void Animator::calculateHierarchicalTransforms(const BoneNode &node,
                                               const glm::mat4 &parentGlobalTransform) {
  // Get local rotation for this bone (identity if not in our transform map)
  glm::mat4 localRotation = glm::mat4(1.0f);
  auto it = boneTransforms.find(node.name);
  if (it != boneTransforms.end()) {
    localRotation = glm::toMat4(it->second.rotation);
  }
  
  // Global transform = parent's global * node's bind pose transform * our rotation
  // The node.transform contains the bind pose transformation from parent to this bone
  glm::mat4 globalTransform = parentGlobalTransform * node.transform * localRotation;
  
  // If this is a bone (has valid ID), update its final matrix
  if (node.boneId >= 0 && node.boneId < (int)finalBoneMatrices.size()) {
    auto &boneInfoMap = model->getBoneInfoMap();
    auto boneIt = boneInfoMap.find(node.name);
    if (boneIt != boneInfoMap.end()) {
      // Final matrix = global transform * inverse bind pose (offset)
      finalBoneMatrices[node.boneId] = globalTransform * boneIt->second.offset;
    }
  }
  
  // Recursively process children - they inherit our global transform
  for (const auto &child : node.children) {
    calculateHierarchicalTransforms(child, globalTransform);
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

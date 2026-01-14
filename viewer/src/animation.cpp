#define GLM_ENABLE_EXPERIMENTAL
#include "animation.h"
#include <glm/gtx/quaternion.hpp>
#include <iostream>

static glm::mat4 convertMatrix(const aiMatrix4x4 &from) {
  return glm::mat4(from.a1, from.b1, from.c1, from.d1, from.a2, from.b2,
                   from.c2, from.d2, from.a3, from.b3, from.c3, from.d3,
                   from.a4, from.b4, from.c4, from.d4);
}

static glm::vec3 getGLMVec(const aiVector3D &vec) {
  return glm::vec3(vec.x, vec.y, vec.z);
}

static glm::quat getGLMQuat(const aiQuaternion &pOrientation) {
  return glm::quat(pOrientation.w, pOrientation.x, pOrientation.y,
                   pOrientation.z);
}

// Bone implementation
Bone::Bone(const std::string &name, int id, const aiNodeAnim *channel)
    : name(name), id(id) {

  // Load positions
  for (unsigned int i = 0; i < channel->mNumPositionKeys; i++) {
    KeyPosition data;
    data.position = getGLMVec(channel->mPositionKeys[i].mValue);
    data.timeStamp = static_cast<float>(channel->mPositionKeys[i].mTime);
    positions.push_back(data);
  }

  // Load rotations
  for (unsigned int i = 0; i < channel->mNumRotationKeys; i++) {
    KeyRotation data;
    data.orientation = getGLMQuat(channel->mRotationKeys[i].mValue);
    data.timeStamp = static_cast<float>(channel->mRotationKeys[i].mTime);
    rotations.push_back(data);
  }

  // Load scales
  for (unsigned int i = 0; i < channel->mNumScalingKeys; i++) {
    KeyScale data;
    data.scale = getGLMVec(channel->mScalingKeys[i].mValue);
    data.timeStamp = static_cast<float>(channel->mScalingKeys[i].mTime);
    scales.push_back(data);
  }
}

void Bone::update(float animationTime) {
  glm::mat4 translation = interpolatePosition(animationTime);
  glm::mat4 rotation = interpolateRotation(animationTime);
  glm::mat4 scale = interpolateScaling(animationTime);
  localTransform = translation * rotation * scale;
}

int Bone::getPositionIndex(float animationTime) {
  for (size_t i = 0; i < positions.size() - 1; i++) {
    if (animationTime < positions[i + 1].timeStamp)
      return static_cast<int>(i);
  }
  return static_cast<int>(positions.size() - 2);
}

int Bone::getRotationIndex(float animationTime) {
  for (size_t i = 0; i < rotations.size() - 1; i++) {
    if (animationTime < rotations[i + 1].timeStamp)
      return static_cast<int>(i);
  }
  return static_cast<int>(rotations.size() - 2);
}

int Bone::getScaleIndex(float animationTime) {
  for (size_t i = 0; i < scales.size() - 1; i++) {
    if (animationTime < scales[i + 1].timeStamp)
      return static_cast<int>(i);
  }
  return static_cast<int>(scales.size() - 2);
}

float Bone::getScaleFactor(float lastTimeStamp, float nextTimeStamp,
                           float animationTime) {
  float midWayLength = animationTime - lastTimeStamp;
  float framesDiff = nextTimeStamp - lastTimeStamp;
  return midWayLength / framesDiff;
}

glm::mat4 Bone::interpolatePosition(float animationTime) {
  if (positions.size() == 1)
    return glm::translate(glm::mat4(1.0f), positions[0].position);

  int p0Index = getPositionIndex(animationTime);
  int p1Index = p0Index + 1;

  if (p1Index >= static_cast<int>(positions.size()))
    return glm::translate(glm::mat4(1.0f), positions[p0Index].position);

  float scaleFactor =
      getScaleFactor(positions[p0Index].timeStamp, positions[p1Index].timeStamp,
                     animationTime);
  glm::vec3 finalPosition = glm::mix(positions[p0Index].position,
                                     positions[p1Index].position, scaleFactor);
  return glm::translate(glm::mat4(1.0f), finalPosition);
}

glm::mat4 Bone::interpolateRotation(float animationTime) {
  if (rotations.size() == 1) {
    auto rotation = glm::normalize(rotations[0].orientation);
    return glm::toMat4(rotation);
  }

  int p0Index = getRotationIndex(animationTime);
  int p1Index = p0Index + 1;

  if (p1Index >= static_cast<int>(rotations.size())) {
    auto rotation = glm::normalize(rotations[p0Index].orientation);
    return glm::toMat4(rotation);
  }

  float scaleFactor =
      getScaleFactor(rotations[p0Index].timeStamp, rotations[p1Index].timeStamp,
                     animationTime);
  glm::quat finalRotation =
      glm::slerp(rotations[p0Index].orientation, rotations[p1Index].orientation,
                 scaleFactor);
  finalRotation = glm::normalize(finalRotation);
  return glm::toMat4(finalRotation);
}

glm::mat4 Bone::interpolateScaling(float animationTime) {
  if (scales.size() == 1)
    return glm::scale(glm::mat4(1.0f), scales[0].scale);

  int p0Index = getScaleIndex(animationTime);
  int p1Index = p0Index + 1;

  if (p1Index >= static_cast<int>(scales.size()))
    return glm::scale(glm::mat4(1.0f), scales[p0Index].scale);

  float scaleFactor = getScaleFactor(scales[p0Index].timeStamp,
                                     scales[p1Index].timeStamp, animationTime);
  glm::vec3 finalScale =
      glm::mix(scales[p0Index].scale, scales[p1Index].scale, scaleFactor);
  return glm::scale(glm::mat4(1.0f), finalScale);
}

// Animation implementation
Animation::Animation(const std::string &animationPath, int animIndex) {
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(
      animationPath, aiProcess_Triangulate | aiProcess_LimitBoneWeights);

  if (!scene || !scene->mRootNode) {
    std::cerr << "ERROR: Failed to load animation file: " << animationPath
              << std::endl;
    return;
  }

  if (scene->mNumAnimations == 0) {
    std::cerr << "WARNING: No animations found in file: " << animationPath
              << std::endl;
    return;
  }

  if (animIndex >= static_cast<int>(scene->mNumAnimations)) {
    animIndex = 0;
  }

  aiAnimation *animation = scene->mAnimations[animIndex];
  name = animation->mName.C_Str();
  duration = static_cast<float>(animation->mDuration);
  ticksPerSecond = static_cast<float>(animation->mTicksPerSecond);

  if (ticksPerSecond == 0.0f) {
    ticksPerSecond = 25.0f;
  }

  std::cout << "Loading animation: " << name << std::endl;
  std::cout << "  Duration: " << duration << " ticks" << std::endl;
  std::cout << "  Ticks/sec: " << ticksPerSecond << std::endl;
  std::cout << "  Channels: " << animation->mNumChannels << std::endl;

  readHierarchyData(rootNode, scene->mRootNode);

  // First pass: collect bone info from the scene
  for (unsigned int meshIdx = 0; meshIdx < scene->mNumMeshes; meshIdx++) {
    aiMesh *mesh = scene->mMeshes[meshIdx];
    for (unsigned int boneIdx = 0; boneIdx < mesh->mNumBones; boneIdx++) {
      std::string boneName = mesh->mBones[boneIdx]->mName.C_Str();
      if (boneInfoMap.find(boneName) == boneInfoMap.end()) {
        BoneInfo info;
        info.id = static_cast<int>(boneInfoMap.size());
        info.offset = convertMatrix(mesh->mBones[boneIdx]->mOffsetMatrix);
        boneInfoMap[boneName] = info;
      }
    }
  }

  // Load animation channels (bones)
  for (unsigned int i = 0; i < animation->mNumChannels; i++) {
    aiNodeAnim *channel = animation->mChannels[i];
    std::string boneName = channel->mNodeName.C_Str();

    // Add bone to map if not present
    if (boneInfoMap.find(boneName) == boneInfoMap.end()) {
      BoneInfo info;
      info.id = static_cast<int>(boneInfoMap.size());
      info.offset = glm::mat4(1.0f);
      boneInfoMap[boneName] = info;
    }

    bones.emplace_back(boneName, boneInfoMap[boneName].id, channel);
  }

  std::cout << "  Loaded " << bones.size() << " bone channels" << std::endl;
}

Bone *Animation::findBone(const std::string &name) {
  for (auto &bone : bones) {
    if (bone.name == name)
      return &bone;
  }
  return nullptr;
}

void Animation::readHierarchyData(AssimpNodeData &dest, const aiNode *src) {
  dest.name = src->mName.C_Str();
  dest.transformation = convertMatrix(src->mTransformation);

  for (unsigned int i = 0; i < src->mNumChildren; i++) {
    AssimpNodeData child;
    readHierarchyData(child, src->mChildren[i]);
    dest.children.push_back(child);
  }
}

int Animation::getAnimationCount(const std::string &path) {
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate);
  if (!scene)
    return 0;
  return static_cast<int>(scene->mNumAnimations);
}

std::vector<std::string> Animation::getAnimationNames(const std::string &path) {
  std::vector<std::string> names;
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate);
  if (!scene)
    return names;

  for (unsigned int i = 0; i < scene->mNumAnimations; i++) {
    names.push_back(scene->mAnimations[i]->mName.C_Str());
  }
  return names;
}

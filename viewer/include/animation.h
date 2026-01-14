#ifndef ANIMATION_H
#define ANIMATION_H

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <string>
#include <vector>

#include "mesh.h"

struct KeyPosition {
  glm::vec3 position;
  float timeStamp;
};

struct KeyRotation {
  glm::quat orientation;
  float timeStamp;
};

struct KeyScale {
  glm::vec3 scale;
  float timeStamp;
};

class Bone {
public:
  std::string name;
  int id;
  glm::mat4 localTransform{1.0f};

  std::vector<KeyPosition> positions;
  std::vector<KeyRotation> rotations;
  std::vector<KeyScale> scales;

  Bone(const std::string &name, int id, const aiNodeAnim *channel);

  void update(float animationTime);
  glm::mat4 getLocalTransform() const { return localTransform; }

  int getPositionIndex(float animationTime);
  int getRotationIndex(float animationTime);
  int getScaleIndex(float animationTime);

private:
  float getScaleFactor(float lastTimeStamp, float nextTimeStamp,
                       float animationTime);
  glm::mat4 interpolatePosition(float animationTime);
  glm::mat4 interpolateRotation(float animationTime);
  glm::mat4 interpolateScaling(float animationTime);
};

struct AssimpNodeData {
  glm::mat4 transformation;
  std::string name;
  std::vector<AssimpNodeData> children;
};

class Animation {
public:
  Animation() = default;
  Animation(const std::string &animationPath, int animIndex = 0);

  Bone *findBone(const std::string &name);

  float getTicksPerSecond() const { return ticksPerSecond; }
  float getDuration() const { return duration; }
  const AssimpNodeData &getRootNode() const { return rootNode; }
  bool isValid() const { return duration > 0.0f && !bones.empty(); }
  const std::map<std::string, BoneInfo> &getBoneIDMap() const {
    return boneInfoMap;
  }
  const std::string &getName() const { return name; }

  static int getAnimationCount(const std::string &path);
  static std::vector<std::string> getAnimationNames(const std::string &path);

private:
  float duration = 0.0f;
  float ticksPerSecond = 25.0f;
  std::string name;
  std::vector<Bone> bones;
  AssimpNodeData rootNode;
  std::map<std::string, BoneInfo> boneInfoMap;

  void readMissingBones(const aiAnimation *animation,
                        std::map<std::string, BoneInfo> &modelBoneInfo);
  void readHierarchyData(AssimpNodeData &dest, const aiNode *src);
};

#endif

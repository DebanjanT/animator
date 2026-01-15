#ifndef ANIMATOR_H
#define ANIMATOR_H

#include "animation.h"
#include "model.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <memory>
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

  // Animation playback
  void loadAnimation(const std::string &path, int animIndex = 0);
  void updateAnimation(float deltaTime);
  void playAnimation(Animation *animation);

  // Playback controls
  void play() { playing = true; }
  void pause() { playing = false; }
  void stop() {
    playing = false;
    currentTime = 0.0f;
  }
  void setSpeed(float speed) { playbackSpeed = speed; }
  void setTime(float time) { currentTime = time; }
  void setLooping(bool loop) { looping = loop; }

  // Getters
  bool isPlaying() const { return playing; }
  float getSpeed() const { return playbackSpeed; }
  float getCurrentTime() const { return currentTime; }
  float getDuration() const;
  float getProgress() const;
  bool isLooping() const { return looping; }
  Animation *getCurrentAnimation() { return currentAnimation; }
  const std::vector<std::string> &getAnimationNames() const {
    return animationNames;
  }

  // Legacy API
  void updateBoneTransform(const std::string &boneName,
                           const BoneTransform &transform);
  void updateBoneTransformByIndex(int boneIndex, const glm::mat4 &transform);
  void
  setBoneTransforms(const std::map<std::string, BoneTransform> &transforms);
  void setFinalBoneMatrices(const std::vector<glm::mat4> &matrices);
  void calculateBoneTransforms();
  std::vector<glm::mat4> &getFinalBoneMatrices() { return finalBoneMatrices; }
  bool hasExternalTransforms() const { return !boneTransforms.empty(); }
  const std::map<std::string, BoneTransform> &getBoneTransforms() const { return boneTransforms; }

  void reset();

  // UE5 Mannequin bone names mapping
  static const std::map<std::string, std::string> UE5_BONE_MAPPING;

private:
  Model *model = nullptr;
  std::vector<glm::mat4> finalBoneMatrices;
  std::map<std::string, BoneTransform> boneTransforms;

  // Animation state
  std::unique_ptr<Animation> ownedAnimation;
  Animation *currentAnimation = nullptr;
  std::vector<std::string> animationNames;
  float currentTime = 0.0f;
  float playbackSpeed = 1.0f;
  bool playing = true;
  bool looping = true;

  void calculateBoneTransform(const AssimpNodeData &node,
                              const glm::mat4 &parentTransform);
  void calculateHierarchicalTransforms(const BoneNode &node,
                                       const glm::mat4 &parentGlobalTransform);
  glm::mat4 getBoneLocalTransform(const std::string &boneName);
};

#endif

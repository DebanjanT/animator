#ifndef SHADER_H
#define SHADER_H

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

class Shader {
public:
  unsigned int ID;

  Shader();
  Shader(const std::string &vertexPath, const std::string &fragmentPath);
  ~Shader();

  void load(const std::string &vertexPath, const std::string &fragmentPath);
  void loadFromSource(const std::string &vertexSource,
                      const std::string &fragmentSource);
  void use() const;

  void setBool(const std::string &name, bool value) const;
  void setInt(const std::string &name, int value) const;
  void setFloat(const std::string &name, float value) const;
  void setVec2(const std::string &name, const glm::vec2 &value) const;
  void setVec3(const std::string &name, const glm::vec3 &value) const;
  void setVec4(const std::string &name, const glm::vec4 &value) const;
  void setMat3(const std::string &name, const glm::mat3 &mat) const;
  void setMat4(const std::string &name, const glm::mat4 &mat) const;
  void setMat4Array(const std::string &name,
                    const std::vector<glm::mat4> &matrices) const;

private:
  mutable std::unordered_map<std::string, int> uniformLocationCache;

  int getUniformLocation(const std::string &name) const;
  void checkCompileErrors(unsigned int shader, const std::string &type);
};

#endif

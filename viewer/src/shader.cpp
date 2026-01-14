#include "shader.h"
#include <fstream>
#include <iostream>
#include <sstream>

Shader::Shader() : ID(0) {}

Shader::Shader(const std::string &vertexPath, const std::string &fragmentPath)
    : ID(0) {
  load(vertexPath, fragmentPath);
}

Shader::~Shader() {
  if (ID != 0) {
    glDeleteProgram(ID);
  }
}

void Shader::load(const std::string &vertexPath,
                  const std::string &fragmentPath) {
  std::string vertexCode, fragmentCode;
  std::ifstream vShaderFile, fShaderFile;

  vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    vShaderFile.open(vertexPath);
    fShaderFile.open(fragmentPath);

    std::stringstream vShaderStream, fShaderStream;
    vShaderStream << vShaderFile.rdbuf();
    fShaderStream << fShaderFile.rdbuf();

    vShaderFile.close();
    fShaderFile.close();

    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();
  } catch (std::ifstream::failure &e) {
    std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what()
              << std::endl;
    return;
  }

  loadFromSource(vertexCode, fragmentCode);
}

void Shader::loadFromSource(const std::string &vertexSource,
                            const std::string &fragmentSource) {
  const char *vShaderCode = vertexSource.c_str();
  const char *fShaderCode = fragmentSource.c_str();

  unsigned int vertex, fragment;

  vertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex, 1, &vShaderCode, NULL);
  glCompileShader(vertex);
  checkCompileErrors(vertex, "VERTEX");

  fragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment, 1, &fShaderCode, NULL);
  glCompileShader(fragment);
  checkCompileErrors(fragment, "FRAGMENT");

  ID = glCreateProgram();
  glAttachShader(ID, vertex);
  glAttachShader(ID, fragment);
  glLinkProgram(ID);
  checkCompileErrors(ID, "PROGRAM");

  glDeleteShader(vertex);
  glDeleteShader(fragment);
}

void Shader::use() const { glUseProgram(ID); }

int Shader::getUniformLocation(const std::string &name) const {
  auto it = uniformLocationCache.find(name);
  if (it != uniformLocationCache.end()) {
    return it->second;
  }
  int location = glGetUniformLocation(ID, name.c_str());
  uniformLocationCache[name] = location;
  return location;
}

void Shader::setBool(const std::string &name, bool value) const {
  glUniform1i(getUniformLocation(name), (int)value);
}

void Shader::setInt(const std::string &name, int value) const {
  glUniform1i(getUniformLocation(name), value);
}

void Shader::setFloat(const std::string &name, float value) const {
  glUniform1f(getUniformLocation(name), value);
}

void Shader::setVec2(const std::string &name, const glm::vec2 &value) const {
  glUniform2fv(getUniformLocation(name), 1, &value[0]);
}

void Shader::setVec3(const std::string &name, const glm::vec3 &value) const {
  glUniform3fv(getUniformLocation(name), 1, &value[0]);
}

void Shader::setVec4(const std::string &name, const glm::vec4 &value) const {
  glUniform4fv(getUniformLocation(name), 1, &value[0]);
}

void Shader::setMat3(const std::string &name, const glm::mat3 &mat) const {
  glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const {
  glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setMat4Array(const std::string &name,
                          const std::vector<glm::mat4> &matrices) const {
  glUniformMatrix4fv(getUniformLocation(name), (GLsizei)matrices.size(),
                     GL_FALSE, &matrices[0][0][0]);
}

void Shader::checkCompileErrors(unsigned int shader, const std::string &type) {
  int success;
  char infoLog[1024];

  if (type != "PROGRAM") {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 1024, NULL, infoLog);
      std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                << infoLog << std::endl;
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 1024, NULL, infoLog);
      std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                << infoLog << std::endl;
    }
  }
}

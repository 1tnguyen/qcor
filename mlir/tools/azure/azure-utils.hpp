#pragma once
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
namespace qcor {
namespace utils {
class Url {
private:
  std::string m_scheme;
  std::string m_host;
  uint16_t m_port{0};
  std::string m_encodedPath;
  std::map<std::string, std::string> m_encodedQueryParameters;
  void AppendQueryParameters(const std::string &encodedQueryParameters);

public:
  explicit Url(const std::string &encodedUrl);
  void SetScheme(const std::string &scheme) { m_scheme = scheme; }
  void SetHost(const std::string &encodedHost) { m_host = encodedHost; }
  void SetPort(uint16_t port) { m_port = port; }
  void SetPath(const std::string &encodedPath) { m_encodedPath = encodedPath; }
  void SetQueryParameters(std::map<std::string, std::string> queryParameters) {
    m_encodedQueryParameters = std::move(queryParameters);
  }
  void AppendPath(const std::string &encodedPath) {
    if (!m_encodedPath.empty() && m_encodedPath.back() != '/') {
      m_encodedPath += '/';
    }
    m_encodedPath += encodedPath;
  }
  void AppendQueryParameter(const std::string &encodedKey,
                            const std::string &encodedValue) {
    m_encodedQueryParameters[encodedKey] = encodedValue;
  }
  void RemoveQueryParameter(const std::string &encodedKey) {
    m_encodedQueryParameters.erase(encodedKey);
  }
  const std::string &GetHost() const { return m_host; }
  const std::string &GetPath() const { return m_encodedPath; }
  uint16_t GetPort() const { return m_port; }
  std::map<std::string, std::string> GetQueryParameters() const {
    return m_encodedQueryParameters;
  }
  const std::string &GetScheme() const { return m_scheme; }
  std::string GetRelativeUrl() const;
  std::string GetAbsoluteUrl() const;
  std::string GetUrlWithoutQuery(bool relative) const;
};
} // namespace utils
} // namespace qcor
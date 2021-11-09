#include "azure-utils.hpp"
#include <algorithm>
#include <cctype>
#include <iterator>
#include <limits>
#include <stdexcept>
namespace {
std::string FormatEncodedUrlQueryParameters(
    std::map<std::string, std::string> const &encodedQueryParameters) {
  {
    std::string queryStr;
    if (!encodedQueryParameters.empty()) {
      auto separator = '?';
      for (const auto &q : encodedQueryParameters) {
        queryStr += separator + q.first + '=' + q.second;
        separator = '&';
      }
    }

    return queryStr;
  }
}
} // namespace

namespace qcor {
namespace utils {
std::string Url::GetUrlWithoutQuery(bool relative) const {
  std::string url;

  if (!relative) {
    if (!m_scheme.empty()) {
      url += m_scheme + "://";
    }
    url += m_host;
    if (m_port != 0) {
      url += ":" + std::to_string(m_port);
    }
  }

  if (!m_encodedPath.empty()) {
    if (!relative) {
      url += "/";
    }

    url += m_encodedPath;
  }

  return url;
}

void Url::AppendQueryParameters(const std::string &query) {
  std::string::const_iterator cur = query.begin();
  if (cur != query.end() && *cur == '?') {
    ++cur;
  }

  while (cur != query.end()) {
    auto key_end = std::find(cur, query.end(), '=');
    std::string query_key = std::string(cur, key_end);

    cur = key_end;
    if (cur != query.end()) {
      ++cur;
    }

    auto value_end = std::find(cur, query.end(), '&');
    std::string query_value = std::string(cur, value_end);

    cur = value_end;
    if (cur != query.end()) {
      ++cur;
    }
    m_encodedQueryParameters[std::move(query_key)] = std::move(query_value);
  }
}

Url::Url(const std::string &url) {
  std::string::const_iterator pos = url.begin();
  const std::string schemeEnd = "://";
  auto schemeIter = url.find(schemeEnd);
  if (schemeIter != std::string::npos) {
    std::transform(url.begin(), url.begin() + schemeIter,
                   std::back_inserter(m_scheme), ::tolower);

    pos = url.begin() + schemeIter + schemeEnd.length();
  }

  auto hostIter = std::find_if(
      pos, url.end(), [](char c) { return c == '/' || c == '?' || c == ':'; });
  m_host = std::string(pos, hostIter);
  pos = hostIter;

  if (pos != url.end() && *pos == ':') {
    auto port_ite = std::find_if_not(pos + 1, url.end(), [](char c) {
      return std::isdigit(static_cast<unsigned char>(c));
    });
    auto portNumber = std::stoi(std::string(pos + 1, port_ite));

    // stoi will throw out_of_range when `int` is overflow, but we need to throw
    // if uint16 is overflow
    auto maxPortNumberSupported = std::numeric_limits<uint16_t>::max();
    if (portNumber > maxPortNumberSupported) {
      throw std::out_of_range(
          "The port number is out of range. The max supported number is " +
          std::to_string(maxPortNumberSupported) + ".");
    }
    // cast is safe because the overflow was detected before
    m_port = static_cast<uint16_t>(portNumber);
    pos = port_ite;
  }

  if (pos != url.end() && (*pos != '/') && (*pos != '?')) {
    // only char `\` or `?` is valid after the port (or the end of the URL). Any
    // other char is an invalid input
    throw std::invalid_argument("The port number contains invalid characters.");
  }

  if (pos != url.end() && (*pos == '/')) {
    auto pathIter = std::find(pos + 1, url.end(), '?');
    m_encodedPath = std::string(pos + 1, pathIter);
    pos = pathIter;
  }

  if (pos != url.end() && *pos == '?') {
    auto queryIter = std::find(pos + 1, url.end(), '#');
    AppendQueryParameters(std::string(pos + 1, queryIter));
    pos = queryIter;
  }
}

std::string Url::GetRelativeUrl() const {
  return GetUrlWithoutQuery(true) +
         FormatEncodedUrlQueryParameters(m_encodedQueryParameters);
}
std::string Url::GetAbsoluteUrl() const {
  return GetUrlWithoutQuery(false) +
         FormatEncodedUrlQueryParameters(m_encodedQueryParameters);
}
} // namespace utils
} // namespace qcor
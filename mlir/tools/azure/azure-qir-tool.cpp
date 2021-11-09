#include "cpr/cpr.h"
#include "azure-utils.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <random>
#include <regex>
namespace {
std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, delim);) {
    *(std::back_inserter(elems))++ = item;
  }
  return elems;
}

std::string trim(const std::string &str) {
  return std::regex_replace(str, std::regex("(^[ ]+)|([ ]+$)"), "");
}

std::string getBaseUrl(const std::string &region,
                       const std::string &subscriptionId,
                       const std::string &resourceGroup,
                       const std::string &workspaceName) {
  const auto formatRegion = [](const std::string &rawRegionString) {
    std::string regionString = rawRegionString;
    // e.g., East US -> eastus
    regionString.erase(
        std::remove_if(regionString.begin(), regionString.end(), isspace),
        regionString.end());
    std::transform(regionString.begin(), regionString.end(),
                   regionString.begin(), ::tolower);
    return regionString;
  };
  return "https://" + formatRegion(region) +
         ".quantum.azure.com/v1.0/subscriptions/" + subscriptionId +
         "/resourceGroups/" + resourceGroup +
         "/providers/Microsoft.Quantum/workspaces/" + workspaceName;
}

std::pair<std::string, std::string>
getConfigInfo(const std::string &configFilePath) {
  std::ifstream configFile(configFilePath);
  const std::string contents((std::istreambuf_iterator<char>(configFile)),
                             std::istreambuf_iterator<char>());
  const auto lines = split(contents, '\n');
  std::string token, location, subscriptionId, resourceGroup, workspaceName;
  for (const auto &line : lines) {
    if (line.find("key") != std::string::npos) {
      token = trim(split(line, ':')[1]);
    }
    if (line.find("Location") != std::string::npos) {
      location = trim(split(line, ':')[1]);
    }
    if (line.find("expires") != std::string::npos) {
      const auto expiration = std::stoll(trim(split(line, ':')[1]));
      if (std::time(nullptr) > expiration) {
        throw std::runtime_error("Token expired");
      }
    }
    if (line.find("Resource ID") != std::string::npos) {
      const auto resourcePath = trim(split(line, ':')[1]);
      const auto components = split(resourcePath, '/');
      for (size_t i = 0; i < components.size(); ++i) {
        if (components[i] == "subscriptions") {
          subscriptionId = components[i + 1];
        }
        if (components[i] == "resourceGroups") {
          resourceGroup = components[i + 1];
        }
        if (components[i] == "Workspaces") {
          workspaceName = components[i + 1];
        }
      }
    }
  }
  assert(!token.empty());
  assert(!subscriptionId.empty());
  assert(!resourceGroup.empty());
  assert(!workspaceName.empty());
  assert(!location.empty());
  return std::make_pair(
      getBaseUrl(location, subscriptionId, resourceGroup, workspaceName),
      token);
}

std::string randomIdStr() {
  static std::random_device dev;
  static std::mt19937 rng(dev());

  std::uniform_int_distribution<int> dist(0, 15);

  const char *v = "0123456789abcdef";
  const bool dash[] = {0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0};

  std::string res;
  for (int i = 0; i < 16; ++i) {
    if (dash[i]) {
      res += "-";
    }
    res += v[dist(rng)];
    res += v[dist(rng)];
  }
  return res;
}

std::pair<std::string, std::string>
createContainerBlob(const qcor::utils::Url &url) {
  auto request = url;
  request.AppendQueryParameter("restype", "container");
  cpr::Header cprHeaders;
  cprHeaders.insert({"Content-Length", "0"});
  cprHeaders.insert({"x-ms-version", "2020-02-10"});
  auto response = cpr::Put(cpr::Url{request.GetAbsoluteUrl()}, cprHeaders,
                           cpr::VerifySsl(false));
  if (response.status_code != 201) {
    throw std::runtime_error("Failed to create container blob");
  }
  return std::make_pair(response.header["ETag"],
                        response.header["x-ms-request-id"]);
}

std::string uploadContainerBlob(const qcor::utils::Url &url,
                                const std::string &blobName,
                                std::stringstream &stream) {
  auto request = url;
  request.AppendPath(blobName);
  cpr::Header cprHeaders;
  const auto requestBody = stream.str();
  cprHeaders.insert({"Content-Length", std::to_string(requestBody.size())});
  cprHeaders.insert({"x-ms-version", "2020-02-10"});
  cprHeaders.insert({"Accept-Encoding", "gzip, deflate"});
  cprHeaders.insert({"Connection", "keep-alive"});
  cprHeaders.insert({"x-ms-blob-type", "BlockBlob"});
  cprHeaders.insert({"Accept", "application/xml"});
  cprHeaders.insert({"x-ms-blob-content-encoding", "gzip"});
  cprHeaders.insert({"x-ms-blob-content-type", "application/json"});
  cprHeaders.insert({"Content-Type", "application/octet-stream"});
  auto response = cpr::Put(cpr::Url{request.GetAbsoluteUrl()}, cprHeaders,
                           cpr::Body{requestBody}, cpr::VerifySsl(false));
  if (response.status_code != 201) {
    throw std::runtime_error("Failed to upload container blob");
  }

  return request.GetUrlWithoutQuery(false);
}
} // namespace

int main(int argc, char **argv) {
  const std::string azureConfigFilename(std::string(getenv("HOME")) +
                                        "/.azure_config");
  // At a high level we need to:
  // 1. Authenticate using AAD
  // This is required to get an access token (in the .azure_config file)
  // "qcor -set-credentials azure"
  if (std::filesystem::exists(azureConfigFilename)) {
    const auto [baseUrl, accessToken] = getConfigInfo(azureConfigFilename);
    // std::cout << "baseUrl: " << baseUrl << std::endl;
    // std::cout << "token: " << accessToken << std::endl;
    cpr::Header cprHeaders;
    cprHeaders.insert({"Content-type", "application/json"});
    cprHeaders.insert({"Connection", "keep-alive"});
    cprHeaders.insert({"Accept", "application/json"});
    cprHeaders.insert({"Authorization", "Bearer " + accessToken});
    // Testing: query provider status
    const std::string path = "/providerStatus";
    cpr::Parameters cprParams;
    auto sasResponse = cpr::Get(cpr::Url{baseUrl + path}, cprHeaders, cprParams,
                                cpr::VerifySsl(false));
    std::cout << "Response:" << sasResponse.text << "\n";
    // 2. Upload the QIR definition to an Azure Storage
    // see
    // https://docs.microsoft.com/en-us/rest/api/storageservices/put-blob-from-url
    // Get the SAS token for the Azure Storage storage account associated with
    // the quantum workspace. The SAS URL can be used to upload job input and/or
    // download job output.
    const std::string storagePath = baseUrl + "/storage/sasUri";
    const std::string jobName = "qcor-" + randomIdStr();
    // POST
    nlohmann::json j;
    j["containerName"] = jobName;
    auto r = cpr::Post(cpr::Url{storagePath}, cprHeaders, cpr::Body{j.dump()},
                       cpr::VerifySsl(false));
    std::cout << "SAS Response:" << r.text << "\n";
    nlohmann::json sasJson = nlohmann::json::parse(r.text);
    const std::string sasUri = sasJson["sasUri"].get<std::string>();
    // PUT the JOB data to the Azure Storage
    // TODO: This should be a QIR file
    const std::string body = "howdy-qcor";

    const auto [eTag, requestId] =
        createContainerBlob(qcor::utils::Url{sasUri});
    std::cout << "Etag:" << eTag << "\n";
    std::stringstream stream;
    stream << body;
    static constexpr const char *blobName = "inputData";
    // Blob URI to construct Azure Quantum job
    const auto blobUri =
        uploadContainerBlob(qcor::utils::Url{sasUri}, blobName, stream);
    std::cout << "Blob access url:" << blobUri << std::endl;

    // 3. Create the job metadata and submit a job request to Azure.
    // https://docs.microsoft.com/en-us/rest/api/azurequantum/dataplane/jobs
  } else {
    std::cerr << "Could not find Azure Quantum configuration file "
              << azureConfigFilename << "\n";
    return 1;
  }
}
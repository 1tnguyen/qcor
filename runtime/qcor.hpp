#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include "AcceleratorBuffer.hpp"
#include <future>


namespace xacc {
class Function;
class AcceleratorBuffer;
}

using namespace xacc;

namespace qcor {

class qpu_handler;


void Initialize(int argc, char** argv);
void Initialize(std::vector<std::string> argv);

// Persist the given function to file, return
// the file name
const std::string persistCompiledCircuit(std::shared_ptr<Function> function);

// Load the compiled circuit from file
std::shared_ptr<Function> loadCompiledCircuit(const std::string &fileName);

// Submit an asynchronous job to the QPU
using HandlerLambda = std::function<void(qpu_handler &)>;
std::future<std::shared_ptr<AcceleratorBuffer>> submit(HandlerLambda &&);

} // namespace qcor

#include "qpu_handler.hpp"


#endif

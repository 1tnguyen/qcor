

#include <string>

namespace qcor {

enum OutputType { MLIR, LLVMMLIR, LLVMIR };

const std::string mlir_compile(const std::string& src_language_type,
                               const std::string& src,
                               const std::string& kernel_name,
                               const OutputType& output_type,
                               bool add_entry_point, int opt_level = 3);

int execute(const std::string& src_language_type, const std::string& src,
            const std::string& kernel_name, int opt_level = 3);
}  // namespace qcor
#include "token_collector_util.hpp"
#include <iostream>
#include <sstream>

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Parse/Parser.h"

using namespace clang;

namespace {

std::string qpu_name = "tnqvm";
int shots = 1024;

class QCORSyntaxHandler : public SyntaxHandler {
public:
  QCORSyntaxHandler() : SyntaxHandler("qcor") {}

  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override {

    // FIXME need way to get backend name from user/command line

    // Get the Diagnostics engine and create a few custom
    // error messgaes
    auto &diagnostics = PP.getDiagnostics();
    auto invalid_no_args = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid quantum kernel - must provide at least a qreg argument.");
    auto invalid_arg_type =
        diagnostics.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                    "Invalid quantum kernel - args can only be "
                                    "qreg, double, or std::vector<double>.");
    auto invalid_qreg_name = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid quantum kernel - could not discover qreg variable name.");

    // Get the function prototype as a string
    SourceManager &sm = PP.getSourceManager();
    auto lo = PP.getLangOpts();
    auto function_prototype =
        Lexer::getSourceText(CharSourceRange(D.getSourceRange(), true), sm, lo)
            .str();

    // Get the Function Type Info from the Declarator,
    // If the function has no arguments, then we throw an error
    const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();
    std::string kernel_name = D.getName().Identifier->getName().str();
    if (!FTI.Params) {
      diagnostics.Report(D.getBeginLoc(), invalid_no_args);
    }

    // Loop over the function arguments and get the
    // buffer name and any program parameter doubles.
    std::vector<std::string> program_parameters;
    std::vector<std::string> bufferNames;
    for (unsigned int ii = 0; ii < FTI.NumParams; ii++) {
      auto &paramInfo = FTI.Params[ii];
      auto ident = paramInfo.Ident;
      auto &decl = FTI.Params[ii].Param;
      auto parm_var_decl = cast<ParmVarDecl>(decl);
      if (parm_var_decl) {
        auto type = parm_var_decl->getType().getCanonicalType().getAsString();
        if (type == "class xacc::internal_compiler::qreg") {
          bufferNames.push_back(ident->getName().str());
        } else if (type == "double") {
          program_parameters.push_back(ident->getName().str());
        } else {
          diagnostics.Report(paramInfo.IdentLoc, invalid_arg_type);
        }
      }
    }

    // If we failed to get the name, then we fail
    if (bufferNames.empty()) {
      diagnostics.Report(D.getBeginLoc(), invalid_qreg_name);
      exit(1);
    }

    // Get Tokens as a string, rewrite code
    // with XACC api calls
    auto kernel_src_and_compiler = qcor::run_token_collector(PP, Toks, function_prototype);
    auto kernel_src = kernel_src_and_compiler.first;
    auto compiler_name = kernel_src_and_compiler.second;

    // std::cout << "HELLO:\n" << kernel_src << "\n";
    // Write new source code in place of the
    // provided quantum code tokens
    OS << "compiler_InitializeXACC(\"" + qpu_name + "\", "+std::to_string(shots)+");\n";
    for (auto &buf : bufferNames) {
      OS << buf << ".setNameAndStore(\"" + buf + "\");\n";
    }
    OS << "auto program = getCompiled(\"" << kernel_name << "\");\n";
    OS << "if (!program) {\n";
    OS << "std::string kernel_src = R\"##(" + kernel_src + ")##\";\n";
    OS << "program = compile(\""+compiler_name+"\", kernel_src.c_str());\n";
    OS << "}\n";
    // OS << "optimize(program);\n";

    OS << "if (__execute) {\n";
    if (!program_parameters.empty()) {
      OS << "double * p = new double[" << program_parameters.size() << "];\n";
      for (unsigned int i = 0; i < program_parameters.size(); i++) {
        OS << "p[" << i << "] = " << program_parameters[i] << ";\n";
      }
    }
    if (bufferNames.size() > 1) {
      OS << "xacc::AcceleratorBuffer * buffers[" << bufferNames.size()
         << "] = {";
      OS << bufferNames[0] << ".results()";
      for (unsigned int k = 1; k < bufferNames.size(); k++) {
        OS << ", " << bufferNames[k] << ".results()";
      }
      OS << "};\n";
      OS << "execute(buffers," << bufferNames.size() << ",program";
    } else {
      OS << "execute(" << bufferNames[0] << ".results(), program";
    }

    if (program_parameters.empty()) {
      OS << ");\n";
    } else {
      OS << ", p);\n";
      OS << "delete[] p;\n";
    }
    OS << "}\n";

    // std::cout << "HELLO:\n" << OS.str() << "\n";
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) override {
    OS << "#include \"xacc_internal_compiler.hpp\"\nusing namespace "
          "xacc::internal_compiler;\n";
  }
};

class DoNothingConsumer : public ASTConsumer {
public:
  bool HandleTopLevelDecl(DeclGroupRef DG) override { return true; }
};

class QCORArgs : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<DoNothingConsumer>();
  }


  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      // Example error handling.
      DiagnosticsEngine &D = CI.getDiagnostics();
      if (args[i] == "-qpu") {
        if (i + 1 >= e) {
          D.Report(D.getCustomDiagID(DiagnosticsEngine::Error,
                                     "missing -qpu argument"));
          return false;
        }
        ++i;
        qpu_name = args[i];
      } else if (args[i] == "-shots") {
          ++i;
          shots = std::stoi(args[i]);
      }
    }
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

} // namespace

static SyntaxHandlerRegistry::Add<QCORSyntaxHandler>
    X("qcor", "qcor quantum kernel syntax handler");

static FrontendPluginRegistry::Add<QCORArgs> XX("qcor-args", "");

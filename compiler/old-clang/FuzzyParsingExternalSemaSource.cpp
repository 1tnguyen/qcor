#include "FuzzyParsingExternalSemaSource.hpp"

#include "IRProvider.hpp"
#include "XACC.hpp"
#include "IRGenerator.hpp"

using namespace clang;

namespace qcor {
namespace compiler {

FuzzyParsingExternalSemaSource::FuzzyParsingExternalSemaSource(
    ASTContext &context)
    : m_Context(context) {
  auto irProvider = xacc::getService<xacc::IRProvider>("gate");
  validInstructions = irProvider->getInstructions();
  validInstructions.push_back("CX");
  auto irgens = xacc::getRegisteredIds<xacc::IRGenerator>();
  for (auto& irg : irgens) {
      validInstructions.push_back(irg);
  }

}

bool FuzzyParsingExternalSemaSource::LookupUnqualified(clang::LookupResult &R,
                                                       clang::Scope *S) {
  DeclarationName Name = R.getLookupName();
  std::string unknownName = Name.getAsString();
  // If this is a valid quantum instruction, tell Clang its
  // all gonna be ok, we got this...
  if (std::find(validInstructions.begin(), validInstructions.end(),
                unknownName) != validInstructions.end()) {

    IdentifierInfo *II = Name.getAsIdentifierInfo();
    SourceLocation Loc = R.getNameLoc();
    auto fdecl = FunctionDecl::Create(
        m_Context, R.getSema().getFunctionLevelDeclContext(), Loc, Loc, Name,
        m_Context.DependentTy, 0, SC_None);

    Stmt *S = new (m_Context) NullStmt(Stmt::EmptyShell());

    fdecl->setBody(S);

    R.addDecl(fdecl);
    return true;
  }
  return false;
}
} // namespace compiler
} // namespace qcor
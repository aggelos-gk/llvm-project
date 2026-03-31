#ifndef LLVM_TRANSFORMS_SCALAR_KAWAHITOZEXTALGORITHM_H
#define LLVM_TRANSFORMS_SCALAR_KAWAHITOZEXTALGORITHM_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class FunctionPass;  // <-- add this

// New Pass Manager pass
class KawahitoZextAlgorithmPass : public PassInfoMixin<KawahitoZextAlgorithmPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// Legacy Pass Manager pass creation
FunctionPass *createKawahitoZextAlgorithmPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_KAWAHITOZEXTALGORITHM_H
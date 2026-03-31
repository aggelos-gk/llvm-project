//===-- AddZextAfterAlloca.h - Lazy zext after operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds zext->trunc patterns after i32 operations when values flow
// to 64-bit contexts. It implements a lazy extension strategy where operations
// remain in 32-bit form and are only extended when necessary (e.g., for i64
// returns, stores, or call arguments).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_ADDZEXTAFTERALLOCA_H
#define LLVM_TRANSFORMS_SCALAR_ADDZEXTAFTERALLOCA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class FunctionPass;

// Legacy Pass Manager
FunctionPass *createAddZextAfterAllocaPass();

// New Pass Manager
struct AddZextAfterAllocaPass : public PassInfoMixin<AddZextAfterAllocaPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_ADDZEXTAFTERALLOCA_H
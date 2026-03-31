//===-- KawahitoZextAlgorithm.cpp - Eliminate redundant zext ------*- C++ -*-===//
//
// Kawahito algorithm — EliminateOneExtend (Fig. 11):
//
//   required = FALSE
//
//   /* DU-chain: AnalyzeUSE */
//   for (I = all instructions that USE the destination operand of SXT):
//     required = AnalyzeUSE(SXT, I, TRUE)
//     if (required) break
//
//   /* UD-chain: AnalyzeDEF */
//   if (!required):
//     for (I = all instructions that DEFINE the source operand of SXT):
//       required = AnalyzeDEF(I)
//       if (required) break
//
//   if (!required) eliminate SXT
//
// AnalyzeUSE(SXT, I, ANALYZE_ARRAY) — Fig. 13 (Case 1 + Case 2, no AnalyzeARRAY):
//   FALSE = extension unnecessary (user doesn't need type S)
//   TRUE  = extension necessary
//
//   Case 1: source operand of I does NOT require type S → return FALSE
//   Case 2: requirement of type S of source operand == dest type of I
//           → walk DU-chain: for each J using dest operand of I:
//               if AnalyzeUSE(SXT, J, ...) return TRUE
//             return FALSE
//   Case 3: AnalyzeARRAY — skipped, return TRUE conservatively
//   Default: return TRUE
//
// AnalyzeDEF(I) — Fig. 12:
//   FALSE = extension unnecessary
//   TRUE  = extension necessary
//
//   Case 1: dest type of I == ShortTy → return FALSE
//   Case 2: dest type == src type of I (same-width)
//           → walk UD-chain: for each J defining src operand of I:
//               if AnalyzeDEF(J) return TRUE
//             return FALSE
//   Default: return TRUE
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/KawahitoZextAlgorithm.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace {

struct KawahitoAnalyzer {
  // The narrow type S — source type of the zext being analyzed (e.g. i32)
  Type *ShortTy = nullptr;
  // The wide type — destination type of the zext (e.g. i64)
  Type *WideTy  = nullptr;

  // Visited sets for cycle detection (USE flag / DEF flag from paper)
  SmallSet<Instruction *, 16> VisitedUSE;
  SmallSet<Instruction *, 16> VisitedDEF;

  // -----------------------------------------------------------------------
  // AnalyzeUSE(SXT, I) — Fig. 13, Case 1 + Case 2 only (no AnalyzeARRAY)
  //
  // Walks FORWARD through users of the zext's result.
  //
  // FALSE = the narrow type S is NOT required at this use → zext unnecessary
  // TRUE  = the narrow type S IS required → zext necessary
  //
  // Case 1: The operand slot of I that receives our value does NOT require S.
  //   → I operates purely on WideTy and doesn't need the value bounded in S.
  //   → return FALSE
  //
  // Case 2: The S-requirement of I's input == the type of I's output.
  //   → I propagates the wide value forward (e.g. add i64, phi i64).
  //   → Walk DU-chain of I's output: if any downstream user needs S → TRUE
  //   → return FALSE if none do
  //
  // Default / Case 3: conservative → return TRUE
  // -----------------------------------------------------------------------
  bool AnalyzeUSE(Instruction *SXT, Instruction *I) {
    // Cycle detection (USE flag)
    if (VisitedUSE.count(I))
      return false;
    VisitedUSE.insert(I);

    // ------------------------------------------------------------------
    // icmp: both operands are WideTy → no slot requires S → Case 1 → FALSE
    //       any operand is ShortTy  → slot requires S → TRUE
    // ------------------------------------------------------------------
    if (auto *ICmp = dyn_cast<ICmpInst>(I)) {
      Value *Op0 = ICmp->getOperand(0);
      Value *Op1 = ICmp->getOperand(1);
      if (Op0->getType() == WideTy && Op1->getType() == WideTy)
        return false; // Case 1: wide comparison, no S requirement
      return true;    // one side is narrow → requires S
    }

    // ------------------------------------------------------------------
    // Binary arithmetic on WideTy (add/sub/mul/and/or/xor/shifts/div/rem):
    //   Input slots are WideTy → no S requirement on input → but
    //   the result propagates forward → Case 2: walk users of result.
    // ------------------------------------------------------------------
    if (auto *BinOp = dyn_cast<BinaryOperator>(I)) {
      if (BinOp->getType() == WideTy) {
        // Case 2: S-requirement propagates through to users of BinOp
        for (User *U : BinOp->users())
          if (auto *J = dyn_cast<Instruction>(U))
            if (AnalyzeUSE(SXT, J))
              return true;
        return false;
      }
      // BinOp produces ShortTy → output requires S → TRUE
      return true;
    }

    // ------------------------------------------------------------------
    // PHI node on WideTy: propagates value forward → Case 2
    // PHI node on ShortTy: output requires S → TRUE
    // ------------------------------------------------------------------
    if (auto *PHI = dyn_cast<PHINode>(I)) {
      if (PHI->getType() == WideTy) {
        for (User *U : PHI->users())
          if (auto *J = dyn_cast<Instruction>(U))
            if (AnalyzeUSE(SXT, J))
              return true;
        return false; // Case 2: none of the downstream users needed S
      }
      return true; // PHI produces ShortTy → requires S
    }

    // ------------------------------------------------------------------
    // Store: storing a WideTy value — the slot type is WideTy, no S req
    //   → Case 1 → FALSE
    // ------------------------------------------------------------------
    if (auto *Store = dyn_cast<StoreInst>(I)) {
      if (Store->getValueOperand()->getType() == WideTy)
        return false; // Case 1
      return true;
    }

    // ------------------------------------------------------------------
    // Return: returning WideTy — no S requirement → Case 1 → FALSE
    // ------------------------------------------------------------------
    if (auto *Ret = dyn_cast<ReturnInst>(I)) {
      if (Ret->getReturnValue() && Ret->getReturnValue()->getType() == WideTy)
        return false; // Case 1
      return true;
    }

    // ------------------------------------------------------------------
    // TruncInst back to ShortTy: this IS consuming S → TRUE
    // TruncInst to other type: walk its users (Case 2)
    // ------------------------------------------------------------------
    if (auto *Trunc = dyn_cast<TruncInst>(I)) {
      if (Trunc->getDestTy() == ShortTy)
        return true; // narrowing to S — requires S
      // Truncating to some other width: propagate forward
      for (User *U : Trunc->users())
        if (auto *J = dyn_cast<Instruction>(U))
          if (AnalyzeUSE(SXT, J))
            return true;
      return false;
    }

    // ------------------------------------------------------------------
    // ZExtInst / SExtInst re-extending from WideTy: input slot is WideTy
    //   → no S requirement → Case 1 → FALSE
    // ------------------------------------------------------------------
    if (auto *ZE = dyn_cast<ZExtInst>(I))
      if (ZE->getSrcTy() == WideTy)
        return false;
    if (auto *SE = dyn_cast<SExtInst>(I))
      if (SE->getSrcTy() == WideTy)
        return false;

    // ------------------------------------------------------------------
    // Call: if the argument slot receiving our value is WideTy → Case 1
    // Conservative fallback if we can't determine → TRUE
    // ------------------------------------------------------------------
    if (auto *Call = dyn_cast<CallInst>(I)) {
      for (unsigned i = 0; i < Call->arg_size(); ++i)
        if (Call->getArgOperand(i)->getType() == WideTy)
          return false; // Case 1: arg slot is wide, no S requirement
      return true;
    }

    // Default / Case 3 (AnalyzeARRAY omitted): conservative
    return true;
  }

  // -----------------------------------------------------------------------
  // AnalyzeDEF(I) — Fig. 12
  //
  // Walks BACKWARD through the UD-chain from SXT's source.
  //
  // FALSE = zext unnecessary (source already zero-extended / wide enough)
  // TRUE  = zext necessary
  // -----------------------------------------------------------------------
  bool AnalyzeDEF(Instruction *I) {
    // Cycle detection (DEF flag)
    if (VisitedDEF.count(I))
      return false;
    VisitedDEF.insert(I);

    Type *DestTy = I->getType();

    // Case 1: I produces the narrow type S → the value lives in S here
    //         The zext is therefore not provably redundant from this side
    if (DestTy == ShortTy)
      return false;

    // Case 2: dest type == src type of I (same-width, e.g. add i64, phi i64)
    //         Walk further back through UD-chain
    bool hasSameTypeOperand = false;
    for (unsigned i = 0; i < I->getNumOperands(); ++i) {
      Value *Op = I->getOperand(i);
      if (Op->getType() != DestTy)
        continue;
      hasSameTypeOperand = true;
      if (auto *J = dyn_cast<Instruction>(Op))
        if (AnalyzeDEF(J))
          return true;
    }
    if (hasSameTypeOperand)
      return false;

    // Default: I has no same-type operand (e.g. I is itself a zext/sext)
    // → value was already extended → zext on top is redundant → TRUE
    return true;
  }

  // -----------------------------------------------------------------------
  // EliminateOneExtend — Fig. 11 (exact order from paper)
  // -----------------------------------------------------------------------
  bool EliminateOneExtend(ZExtInst *ZExt) {
    VisitedUSE.clear();
    VisitedDEF.clear();

    Value *Src   = ZExt->getOperand(0);
    Type  *SrcTy = Src->getType();     // ShortTy S, e.g. i32
    Type  *DstTy = ZExt->getDestTy();  // WideTy,   e.g. i64

    ShortTy = SrcTy;
    WideTy  = DstTy;

    bool Required = false;

    // ------------------------------------------------------------------
    // Step 1 (Fig. 11): DU-chain — AnalyzeUSE
    // "for I = all instructions that use the destination operand of SXT"
    // ------------------------------------------------------------------
    for (User *U : ZExt->users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        Required = AnalyzeUSE(ZExt, I);
        if (Required) break;
      }
    }

    // ------------------------------------------------------------------
    // Step 2 (Fig. 11): UD-chain — AnalyzeDEF (only if still not required)
    // "for I = all instructions that define the source operand of SXT"
    // In SSA: exactly one defining instruction (or arg / constant).
    // ------------------------------------------------------------------
    if (!Required) {
      if (auto *SrcInst = dyn_cast<Instruction>(Src)) {
        Required = AnalyzeDEF(SrcInst);
      } else if (isa<Argument>(Src)) {
        // Argument: unknown value from caller — conservative, treat as required
        Required = true;
      } else if (isa<ConstantInt>(Src) || isa<UndefValue>(Src) ||
                 isa<PoisonValue>(Src)) {
        Required = false; // constants never need extension
      }
    }

    if (Required)
      return false; // keep the zext

    // ------------------------------------------------------------------
    // Step 3: Eliminate.
    //
    // Users fall into three buckets:
    //   TruncToSrc  : trunc(zext(x)) to SrcTy   → replace with Src
    //   TruncToOther: trunc(zext(x)) to T≠SrcTy → replace with trunc(Src,T)
    //   OtherUsers  : still need DstTy (i64)     → rebuild zext from Src
    //                 (AnalyzeUSE confirmed they don't need S, so the new
    //                  zext is semantically identical — trunc waste removed)
    // ------------------------------------------------------------------
    SmallVector<TruncInst *, 4> TruncToSrcUsers;
    SmallVector<TruncInst *, 4> TruncToOtherUsers;
    SmallVector<Instruction *, 4> OtherUsers;

    for (User *U : ZExt->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI) continue;
      if (auto *Trunc = dyn_cast<TruncInst>(UI)) {
        if (Trunc->getDestTy() == SrcTy)
          TruncToSrcUsers.push_back(Trunc);
        else
          TruncToOtherUsers.push_back(Trunc);
      } else {
        OtherUsers.push_back(UI);
      }
    }

    // trunc(zext(x)) → x
    for (auto *Trunc : TruncToSrcUsers) {
      Trunc->replaceAllUsesWith(Src);
      Trunc->eraseFromParent();
    }

    // trunc(zext(x), T) → trunc(x, T)
    for (auto *Trunc : TruncToOtherUsers) {
      IRBuilder<> Builder(Trunc);
      Value *New = Builder.CreateTrunc(Src, Trunc->getDestTy(),
                                       Trunc->getName());
      Trunc->replaceAllUsesWith(New);
      Trunc->eraseFromParent();
    }

    if (!OtherUsers.empty()) {
      IRBuilder<> Builder(ZExt);
      Value *NewZExt = Builder.CreateZExt(Src, DstTy, ZExt->getName());
      ZExt->replaceAllUsesWith(NewZExt);
      ZExt->eraseFromParent();
      return !TruncToSrcUsers.empty() || !TruncToOtherUsers.empty();
    }

    // No other users — erase cleanly
    ZExt->eraseFromParent();
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Legacy Pass Manager
//===----------------------------------------------------------------------===//
class KawahitoZextAlgorithm : public FunctionPass {
public:
  static char ID;
  KawahitoZextAlgorithm() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Changed = false;
    SmallVector<ZExtInst *, 32> ZExts;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *ZExt = dyn_cast<ZExtInst>(&I))
          ZExts.push_back(ZExt);

    KawahitoAnalyzer Analyzer;
    for (auto *ZExt : ZExts)
      if (ZExt->getParent())
        if (Analyzer.EliminateOneExtend(ZExt))
          Changed = true;

    return Changed;
  }
};

} // namespace

char KawahitoZextAlgorithm::ID = 0;

INITIALIZE_PASS(KawahitoZextAlgorithm, "kawahito-zext",
                "Eliminate redundant zext using Kawahito algorithm", false, false)

namespace llvm {
FunctionPass *createKawahitoZextAlgorithmPass() {
  return new KawahitoZextAlgorithm();
}
} // namespace llvm

//===----------------------------------------------------------------------===//
// New Pass Manager
//===----------------------------------------------------------------------===//
PreservedAnalyses KawahitoZextAlgorithmPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  bool Changed = false;
  SmallVector<ZExtInst *, 32> ZExts;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *ZExt = dyn_cast<ZExtInst>(&I))
        ZExts.push_back(ZExt);

  KawahitoAnalyzer Analyzer;
  for (auto *ZExt : ZExts)
    if (ZExt->getParent())
      if (Analyzer.EliminateOneExtend(ZExt))
        Changed = true;

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
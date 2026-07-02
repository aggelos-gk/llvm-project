#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Scalar/KawahitoZextAlgorithm.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {

class EliminateRedundantExtend {
public:
  explicit EliminateRedundantExtend(Function &F, BlockFrequencyInfo &BFI)
      : F(F), DL(F.getParent()->getDataLayout()), BFI(BFI) {}

  bool run() {
    std::vector<std::pair<BasicBlock*, uint64_t>> blocks = getBasicBlocksHotness();
    std::vector<CastInst *> Exts = getExtendInstructions(blocks);
    bool Changed = false;

    for (CastInst *Ext : Exts) {
      if (!Ext->getParent())
        continue;

      Changed |= EliminateOneExtend(Ext);
    }

    return Changed;
  }


private:
  Function &F;
  const DataLayout &DL;
  BlockFrequencyInfo &BFI;

  struct AnalyzeState {
    DenseMap<const Value *, DenseSet<const Instruction *>> USE;
  };


  std::vector<std::pair<BasicBlock*, uint64_t>> getBasicBlocksHotness() {
    std::vector<std::pair<BasicBlock*, uint64_t>> blocks;

    for (BasicBlock &BB : F) {
      uint64_t freq = BFI.getBlockFreq(&BB).getFrequency();
      blocks.push_back({&BB, freq});
    }

    std::sort(blocks.begin(), blocks.end(),
          [](const auto &a, const auto &b) {
            return a.second > b.second;
          });

    return blocks;
  }

  bool TryConvertZExtToSExt(ZExtInst *ZExt) {
    auto *SrcTy = dyn_cast<IntegerType>(ZExt->getSrcTy());
    auto *DstTy = dyn_cast<IntegerType>(ZExt->getDestTy());
    if (!SrcTy || !DstTy)
      return false;

    unsigned SrcBits = SrcTy->getBitWidth();
    if (SrcBits == 0)
      return false;

    Value *Src = ZExt->getOperand(0);

    KnownBits KB = computeKnownBits(Src, DL, 0, nullptr, ZExt, nullptr);
    if (!KB.Zero[SrcBits - 1])
      return false;

    IRBuilder<> B(ZExt);
    Value *SExt = B.CreateSExt(Src, DstTy, ZExt->getName() + ".sext");

    ZExt->replaceAllUsesWith(SExt);
    ZExt->eraseFromParent();
    return true;
  }

  bool ConvertZExtToSExt(ZExtInst *ZExt) {
    auto *DstTy = dyn_cast<IntegerType>(ZExt->getDestTy());
    if (!DstTy)
      return false;

    IRBuilder<> B(ZExt);
    Value *SExt =
        B.CreateSExt(ZExt->getOperand(0), DstTy, ZExt->getName() + ".sext");
    ZExt->replaceAllUsesWith(SExt);
    ZExt->eraseFromParent();
    return true;
  }

  bool RemoveCanonicalExtend(CastInst *Ext) {
    auto *Trunc = dyn_cast<TruncInst>(Ext->getOperand(0));
    if (!Trunc)
      return false;

    auto *WideTy = dyn_cast<IntegerType>(Ext->getType());
    auto *SrcTy = dyn_cast<IntegerType>(Ext->getSrcTy());
    auto *TruncSrcTy = dyn_cast<IntegerType>(Trunc->getOperand(0)->getType());
    auto *TruncDstTy = dyn_cast<IntegerType>(Trunc->getType());
    if (!WideTy || !SrcTy || !TruncSrcTy || !TruncDstTy)
      return false;

    if (TruncSrcTy != WideTy || TruncDstTy != SrcTy)
      return false;

    if (SrcTy->getBitWidth() >= WideTy->getBitWidth())
      return false;

    // Collapse the canonical helper form:
    //   %wide = <iM def>
    //   %trunc = trunc iM %wide to iN
    //   %ext = zext/sext iN %trunc to iM

    Value *Wide = Trunc->getOperand(0);
    Ext->replaceAllUsesWith(Wide);
    RecursivelyDeleteTriviallyDeadInstructions(Ext);
    return true;
  }

  bool FoldPreviousSameOpcodeExtend(CastInst *Ext) {
    auto *Trunc = dyn_cast<TruncInst>(Ext->getOperand(0));
    if (!Trunc)
      return false;

    auto *PrevExt = dyn_cast<CastInst>(Trunc->getOperand(0));
    if (!PrevExt)
      return false;

    if (Ext->getOpcode() != PrevExt->getOpcode())
      return false;

    auto *ExtSrcTy = dyn_cast<IntegerType>(Ext->getSrcTy());
    auto *ExtDstTy = dyn_cast<IntegerType>(Ext->getDestTy());
    auto *PrevSrcTy = dyn_cast<IntegerType>(PrevExt->getSrcTy());
    auto *PrevDstTy = dyn_cast<IntegerType>(PrevExt->getDestTy());
    auto *TruncSrcTy = dyn_cast<IntegerType>(Trunc->getSrcTy());
    auto *TruncDstTy = dyn_cast<IntegerType>(Trunc->getDestTy());
    if (!ExtSrcTy || !ExtDstTy || !PrevSrcTy || !PrevDstTy || !TruncSrcTy ||
        !TruncDstTy)
      return false;

    if (PrevSrcTy != ExtSrcTy || PrevDstTy != ExtDstTy)
      return false;

    if (TruncSrcTy != PrevDstTy || TruncDstTy != PrevSrcTy)
      return false;

    Ext->replaceAllUsesWith(PrevExt);
    RecursivelyDeleteTriviallyDeadInstructions(Ext);
    return true;
  }

  bool FoldRoundTripTruncUsers(CastInst *Ext) {
    auto *SrcTy = dyn_cast<IntegerType>(Ext->getSrcTy());
    auto *DstTy = dyn_cast<IntegerType>(Ext->getDestTy());
    if (!SrcTy || !DstTy)
      return false;

    SmallVector<TruncInst *, 8> CandidateTruncs;
    for (User *U : Ext->users()) {
      auto *Trunc = dyn_cast<TruncInst>(U);
      if (!Trunc)
        continue;

      auto *TruncSrcTy = dyn_cast<IntegerType>(Trunc->getSrcTy());
      auto *TruncDstTy = dyn_cast<IntegerType>(Trunc->getDestTy());
      if (!TruncSrcTy || !TruncDstTy)
        continue;

      if (TruncSrcTy == DstTy && TruncDstTy == SrcTy)
        CandidateTruncs.push_back(Trunc);
    }

    bool Changed = false;
    for (TruncInst *Trunc : CandidateTruncs) {
      if (!Trunc->getParent())
        continue;

      Value *Src = Ext->getOperand(0);
      Trunc->replaceAllUsesWith(Src);
      Changed = true;
      RecursivelyDeleteTriviallyDeadInstructions(Trunc);
    }

    if (Changed && Ext->getParent())
      RecursivelyDeleteTriviallyDeadInstructions(Ext);
    return Changed;
  }



  static bool isCandidateExtend(CastInst *Ext) {
    if (!isa<ZExtInst>(Ext) && !isa<SExtInst>(Ext))
      return false;

    auto *SrcTy = dyn_cast<IntegerType>(Ext->getSrcTy());
    auto *DstTy = dyn_cast<IntegerType>(Ext->getDestTy());
    if (!SrcTy || !DstTy)
      return false;

    return SrcTy->getBitWidth() < DstTy->getBitWidth();
  }

  std::vector<CastInst *> getExtendInstructions(
    const std::vector<std::pair<BasicBlock*, uint64_t>> &blocks) {
    std::vector<CastInst *> Exts;

    for (auto &[BB, freq] : blocks) {
      for (Instruction &I : *BB) {
        if (auto *Ext = dyn_cast<CastInst>(&I)) {
          if (isCandidateExtend(Ext))
            Exts.push_back(Ext);
        }
      }
    }

    return Exts;
  }

  static bool areUpperBitsKnownZero(Value *V, unsigned RootBits,
                                    const DataLayout &DL,
                                    Instruction *Ctx) {
    auto *IntTy = dyn_cast<IntegerType>(V->getType());
    if (!IntTy)
      return false;

    unsigned BitWidth = IntTy->getBitWidth();
    if (RootBits >= BitWidth)
      return false;

    KnownBits KB = computeKnownBits(V, DL, 0, nullptr, Ctx, nullptr);
    for (unsigned Bit = RootBits; Bit < BitWidth; ++Bit) {
      if (!KB.Zero[Bit])
        return false;
    }

    return true;
  }

  static bool areUpperBitsKnownOne(Value *V, unsigned RootBits,
                                   const DataLayout &DL,
                                   Instruction *Ctx) {
    auto *IntTy = dyn_cast<IntegerType>(V->getType());
    if (!IntTy)
      return false;

    unsigned BitWidth = IntTy->getBitWidth();
    if (RootBits >= BitWidth)
      return false;

    KnownBits KB = computeKnownBits(V, DL, 0, nullptr, Ctx, nullptr);
    for (unsigned Bit = RootBits; Bit < BitWidth; ++Bit) {
      if (!KB.One[Bit])
        return false;
    }

    return true;
  }

  bool hasZeroUpperBitsForExtend(CastInst *Ext, Value *V,
                                 Instruction *Ctx) const {
    auto *WideTy = dyn_cast<IntegerType>(Ext->getType());
    auto *ValueTy = dyn_cast<IntegerType>(V->getType());
    if (!WideTy || !ValueTy || ValueTy != WideTy)
      return false;

    auto *RootTy = cast<IntegerType>(Ext->getSrcTy());
    return areUpperBitsKnownZero(V, RootTy->getBitWidth(), DL, Ctx);
  }

  bool isUseSink(CastInst *Ext, Value *Current, Instruction *I) {
    auto *SrcTy = dyn_cast<IntegerType>(Ext->getSrcTy());
    if (!SrcTy)
      return false; // conservative

    unsigned RootBits = SrcTy->getBitWidth();

    // Case 1:
    // valid trunc sink means this use-path does not require the zext.
    if (auto *Trunc = dyn_cast<TruncInst>(I)) {
      auto *DstTy = dyn_cast<IntegerType>(Trunc->getDestTy());
      if (!DstTy)
        return false;

      if (DstTy->getBitWidth() <= RootBits)
        return true;
    }

    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      Value *Other = nullptr;
      if (BO->getOperand(0) == Current && BO->getOperand(1) != Current)
        Other = BO->getOperand(1);
      else if (BO->getOperand(1) == Current && BO->getOperand(0) != Current)
        Other = BO->getOperand(0);

      if (Other) {
        switch (BO->getOpcode()) {
        case Instruction::And:
          if (areUpperBitsKnownZero(Other, RootBits, DL, I))
            return true;
          break;
        case Instruction::Or:
          if (areUpperBitsKnownOne(Other, RootBits, DL, I))
            return true;
          break;
        default:
          break;
        }
      }
    }

    return false;
  }


  static bool isCase2Instruction(Instruction *I) {
    if (isa<PHINode>(I))
      return true;

    if (isa<SelectInst>(I))
      return true;

    if (isa<GetElementPtrInst>(I))
      return true;

    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      // Case 2 only recurses through ops whose low RootBits remain a function of
      // the incoming low RootBits alone. Comparisons, right shifts, and
      // division/remainder do not satisfy that property generically in LLVM IR.
      // Flagged mul is also excluded: replacing zext with sext can change LLVM
      // poison semantics even when later low-bit uses look safe.
      switch (BO->getOpcode()) {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
        case Instruction::Shl:
          return true;
        default:
          return false;
      }
    }

    return false;
  }

  static bool isCase2InstructionNNeg(Instruction *I) {
    if (isCase2Instruction(I))
      return true;

    if (isa<ICmpInst>(I))
      return true;

    if (isa<ZExtInst>(I))
      return true;

    if (isa<SExtInst>(I))
      return true;

    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      switch (BO->getOpcode()) {
      case Instruction::Mul:
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::URem:
      case Instruction::SRem:
        return true;
      default:
        return false;
      }
    }

    return false;
  }

  bool AnalyzeUSE(CastInst *Ext, Value *Current, Instruction *I,
                  AnalyzeState &State) {
    // USE flag: already visited for this current path value.
    DenseSet<const Instruction *> &VisitedUsers = State.USE[Current];
    if (!VisitedUsers.insert(I).second)
      return false;

    // Case 1:
    // current use does not require zext
    if (isUseSink(Ext, Current, I))
      return false;

    // Case 2:
    // recurse on all users of I
    bool RelaxedNNeg = false;
    if (auto *ZExt = dyn_cast<ZExtInst>(Ext))
      RelaxedNNeg = ZExt->hasNonNeg();
    if ((RelaxedNNeg && isCase2InstructionNNeg(I)) ||
        (!RelaxedNNeg && isCase2Instruction(I))) {
      for (User *U : I->users()) {
        auto *UserI = dyn_cast<Instruction>(U);
        if (!UserI)
          return true;

        if (AnalyzeUSE(Ext, I, UserI, State))
          return true;
      }

      return false;
    }

    return true;
  }

			  bool EliminateOneExtend(CastInst *Ext) {
        WeakTrackingVH ExtHandle(Ext);
        bool Changed = FoldPreviousSameOpcodeExtend(Ext);
        if (!ExtHandle)
          return true;
        Ext = cast<CastInst>(ExtHandle.operator Value *());

        Changed |= FoldRoundTripTruncUsers(Ext);
        if (!ExtHandle)
          return true;
        Ext = cast<CastInst>(ExtHandle.operator Value *());

		    AnalyzeState State;
		    bool Required = false;

    for (User *U : Ext->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI) {
        Required = true;
        break;
      }

      Required = AnalyzeUSE(Ext, Ext, UI, State);
      if (Required)
        break;
    }

	    if (!Required) {
	      if (RemoveCanonicalExtend(Ext))
	        return true;
	      if (auto *ZExt = dyn_cast<ZExtInst>(Ext))
	        return ConvertZExtToSExt(ZExt) || Changed;
	      return Changed;
	    }

	    return Changed;
	  }
};

} // namespace

PreservedAnalyses KawahitoZextAlgorithmPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  
  BlockFrequencyInfo &BFI = AM.getResult<BlockFrequencyAnalysis>(F);                                               
  EliminateRedundantExtend Pass(F, BFI);
  bool Changed = Pass.run();

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

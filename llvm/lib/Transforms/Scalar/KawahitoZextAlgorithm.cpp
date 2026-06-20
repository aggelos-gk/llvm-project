


#include "llvm/Support/raw_ostream.h"
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
#include "llvm/Transforms/Scalar/KawahitoZextAlgorithm.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {

class EliminateRedundantZext {
public:
  explicit EliminateRedundantZext(Function &F, BlockFrequencyInfo &BFI)
      : F(F), DL(F.getParent()->getDataLayout()), BFI(BFI) {}

  bool run() {
    std::vector<std::pair<BasicBlock*, uint64_t>> blocks = getBasicBlocksHotness();
    std::vector<ZExtInst*> zexts = getZExtInstructions(blocks);
    bool Changed = false;

    print_logs(blocks, zexts);

    for (ZExtInst *ZXT : zexts) {
      if (!ZXT->getParent())
        continue;

      // if (TryConvertZExtToSExt(ZXT)) {
      //   Changed = true;
      //   continue;
      // }

      Changed |= EliminateOneExtend(ZXT);
    }

    return Changed;
  }


private:
  Function &F;
  const DataLayout &DL;
  BlockFrequencyInfo &BFI;

  struct AnalyzeState {
    DenseSet<std::pair<const Instruction *, const Value *>> USE;
    DenseSet<const Instruction*> DEF;
    bool SawGEP = false;
    bool SawTruncSink = false;
  };

  struct BuildState {
    DenseMap<Value *, Value *> Cache;
  };

  static bool isDEFChainOpcode(unsigned Opcode) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      return true;
    default:
      return false;
    }
  }


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



  bool isCandidateZext(ZExtInst *ZExt) {
    Type *DstTy = ZExt->getType();
    return DstTy->isIntegerTy(64);
  }

  std::vector<ZExtInst*> getZExtInstructions(
    const std::vector<std::pair<BasicBlock*, uint64_t>> &blocks) {
    std::vector<ZExtInst*> zexts;

    for (auto &[BB, freq] : blocks) {
      for (Instruction &I : *BB) {
        if (auto *ZExt = dyn_cast<ZExtInst>(&I)) {
          if (isCandidateZext(ZExt))
          zexts.push_back(ZExt);
        }
      }
    }

    return zexts;
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

  bool hasZeroUpperBitsForZExt(ZExtInst *ZExt, Value *V,
                               Instruction *Ctx) const {
    auto *WideTy = dyn_cast<IntegerType>(ZExt->getType());
    auto *ValueTy = dyn_cast<IntegerType>(V->getType());
    if (!WideTy || !ValueTy || ValueTy != WideTy)
      return false;

    auto *RootTy = cast<IntegerType>(ZExt->getSrcTy());
    return areUpperBitsKnownZero(V, RootTy->getBitWidth(), DL, Ctx);
  }

  enum class UseSinkKind {
    RequiresZExt,
    Garbage,
    ZS,
  };

  UseSinkKind classifyUseSink(ZExtInst *ZExt, Value *CurrentValue,
                              Instruction *I) {
    auto *SrcTy = dyn_cast<IntegerType>(ZExt->getSrcTy());
    if (!SrcTy)
      return UseSinkKind::RequiresZExt; // conservative

    unsigned RootBits = SrcTy->getBitWidth();

    // Case 1:
    // valid trunc sink means this use-path does not require the zext.
    if (auto *Trunc = dyn_cast<TruncInst>(I)) {
      auto *DstTy = dyn_cast<IntegerType>(Trunc->getDestTy());
      if (!DstTy)
        return UseSinkKind::RequiresZExt;

      if (DstTy->getBitWidth() <= RootBits)
        return UseSinkKind::Garbage;
    }

    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      Value *Other = nullptr;
      if (BO->getOperand(0) == CurrentValue && BO->getOperand(1) != CurrentValue)
        Other = BO->getOperand(1);
      else if (BO->getOperand(1) == CurrentValue && BO->getOperand(0) != CurrentValue)
        Other = BO->getOperand(0);

      if (Other) {
        switch (BO->getOpcode()) {
        case Instruction::And:
          if (areUpperBitsKnownZero(Other, RootBits, DL, I))
            return UseSinkKind::ZS;
          break;
        case Instruction::Or:
          if (areUpperBitsKnownOne(Other, RootBits, DL, I))
            return UseSinkKind::ZS;
          break;
        default:
          break;
        }
      }
    }

    return UseSinkKind::RequiresZExt;
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

  bool AnalyzeUSE(ZExtInst *ZExt, Instruction *I, AnalyzeState &State) {
    // USE flag: already visited for this ZExt
    return AnalyzeUSE(ZExt, ZExt, I, State);
  }

  bool AnalyzeUSE(ZExtInst *ZExt, Value *CurrentValue, Instruction *I,
                  AnalyzeState &State) {
    if (!State.USE.insert({I, CurrentValue}).second)
      return false;

    if (isa<GetElementPtrInst>(I))
      State.SawGEP = true;

    // Case 1:
    // current use does not require zext
    switch (classifyUseSink(ZExt, CurrentValue, I)) {
    case UseSinkKind::Garbage:
      State.SawTruncSink = true;
      return false;
    case UseSinkKind::ZS:
      return false;
    case UseSinkKind::RequiresZExt:
      break;
    }

    // Case 2:
    // recurse on all users of I
    if (isCase2Instruction(I)) {
      for (User *U : I->users()) {
        auto *UserI = dyn_cast<Instruction>(U);
        if (!UserI)
          return true; // conservative

        if (AnalyzeUSE(ZExt, I, UserI, State))
          return true;
      }

      return false;
    }

    // Default:
    // this use still requires the zext
    return true;
  }

/*
  bool rewriteToNarrow(ZExtInst *ZExt) {
  Type *RootTy = ZExt->getSrcTy();

  DenseMap<Value *, Value *> NarrowMap;
  SmallVector<Instruction *, 16> Worklist;
  SmallVector<Instruction *, 16> Dead;

  NarrowMap[ZExt] = ZExt->getOperand(0);

  auto getNarrow = [&](Value *V, IRBuilder<> &B) -> Value * {
    if (Value *N = NarrowMap.lookup(V))
      return N;

    if (Constant *C = dyn_cast<Constant>(V))
      return ConstantExpr::getTruncOrBitCast(C, RootTy);

    return nullptr;
  };

  auto getNarrowIncoming = [&](Value *V) -> Value * {
    if (Value *N = NarrowMap.lookup(V))
      return N;

    if (Constant *C = dyn_cast<Constant>(V))
      return ConstantExpr::getTruncOrBitCast(C, RootTy);

    return nullptr;
  };

  for (User *U : ZExt->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U))
      Worklist.push_back(I);
    else
      return false;
  }

  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();

    if (auto *TI = dyn_cast<TruncInst>(I)) {
      Value *Narrow = NarrowMap.lookup(TI->getOperand(0));
      if (!Narrow)
        return false;

      Value *Replacement = Narrow;

      if (TI->getDestTy() != Narrow->getType()) {
        IRBuilder<> B(TI);
        Replacement = B.CreateTrunc(Narrow, TI->getDestTy());
      }

      TI->replaceAllUsesWith(Replacement);
      Dead.push_back(TI);
      continue;
    }

    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      IRBuilder<> B(BO);

      Value *L = getNarrow(BO->getOperand(0), B);
      Value *R = getNarrow(BO->getOperand(1), B);

      if (!L || !R)
        return false;

      Value *N = B.CreateBinOp(BO->getOpcode(), L, R);

      NarrowMap[BO] = N;
      Dead.push_back(BO);

      for (User *U : BO->users()) {
        if (Instruction *UserI = dyn_cast<Instruction>(U))
          Worklist.push_back(UserI);
        else
          return false;
      }

      continue;
    }

    if (auto *PN = dyn_cast<PHINode>(I)) {
      PHINode *NewPhi =
          PHINode::Create(RootTy, PN->getNumIncomingValues(), "", PN);

      NarrowMap[PN] = NewPhi;
      Dead.push_back(PN);

      for (unsigned k = 0; k < PN->getNumIncomingValues(); ++k) {
        Value *In = PN->getIncomingValue(k);
        BasicBlock *Pred = PN->getIncomingBlock(k);

        Value *NIn = getNarrowIncoming(In);
        if (!NIn)
          return false;

        NewPhi->addIncoming(NIn, Pred);
      }

      for (User *U : PN->users()) {
        if (Instruction *UserI = dyn_cast<Instruction>(U))
          Worklist.push_back(UserI);
        else
          return false;
      }

      continue;
    }

    return false;
  }

  for (Instruction *I : reverse(Dead)) {
    if (I->use_empty())
      I->eraseFromParent();
  }

  if (ZExt->use_empty())
    ZExt->eraseFromParent();

  return true;
} */
/*
	bool AnalyzeDEFValue(ZExtInst *ZExt, Value *V, AnalyzeState &State) {
    auto *RootTy = cast<IntegerType>(ZExt->getSrcTy());

    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      if (CI->getType() == RootTy)
        return false;

      return !hasZeroUpperBitsForZExt(ZExt, CI, nullptr);
    }

    if (!isa<Instruction>(V))
      return true;

    return AnalyzeDEF(ZExt, cast<Instruction>(V), State);
  }


 bool AnalyzeDEF(ZExtInst *ZExt, Instruction *I, AnalyzeState &State) {
    if (!State.DEF.insert(I).second) {
      return false;
    }

    auto *RootTy = cast<IntegerType>(ZExt->getSrcTy());

    // Case 1:
	    if (auto *PrevZExt = dyn_cast<ZExtInst>(I)) {
	      if (PrevZExt->getType() == ZExt->getType())
	        return false;

        return AnalyzeDEFValue(ZExt, PrevZExt->getOperand(0), State);
	    }

	    if (auto *TI = dyn_cast<TruncInst>(I)) {
	      if (hasZeroUpperBitsForZExt(ZExt, TI->getOperand(0), TI))
	        return false;

	      return AnalyzeDEFValue(ZExt, TI->getOperand(0), State);
	    }

    if (auto *SExt = dyn_cast<SExtInst>(I)) {
      if (SExt->getType() == ZExt->getType() &&
          SExt->getSrcTy() == RootTy &&
          hasZeroUpperBitsForZExt(ZExt, SExt, SExt))
        return false;

      return AnalyzeDEFValue(ZExt, SExt->getOperand(0), State);
    }

	    if (auto *PN = dyn_cast<PHINode>(I)) {
        if (PN->getType() == ZExt->getType() &&
            hasZeroUpperBitsForZExt(ZExt, PN, PN))
          return false;

	      for (unsigned k = 0; k < PN->getNumIncomingValues(); ++k) {
	        if (AnalyzeDEFValue(ZExt, PN->getIncomingValue(k), State))
	          return true;
      }
      return false;
    }

	    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        auto *WideTy = cast<IntegerType>(ZExt->getType());
        if (BO->getType() != WideTy || !isDEFChainOpcode(BO->getOpcode()))
          return true;

        if (hasZeroUpperBitsForZExt(ZExt, BO, BO))
          return false;

        if (AnalyzeDEFValue(ZExt, BO->getOperand(0), State))
          return true;
        if (AnalyzeDEFValue(ZExt, BO->getOperand(1), State))
          return true;
        return false;
	    }

	    return true;
	  }

	  Value *BuildDEFReplacementValue(ZExtInst *ZExt, Value *V, BuildState &State) {
	    if (Value *Cached = State.Cache.lookup(V))
	      return Cached;

	    auto *RootTy = cast<IntegerType>(ZExt->getSrcTy());
	    auto *WideTy = cast<IntegerType>(ZExt->getType());

	    if (auto *CI = dyn_cast<ConstantInt>(V)) {
        if (CI->getType() == RootTy)
          return ConstantInt::get(WideTy,
                                  CI->getValue().zext(WideTy->getBitWidth()));
        if (hasZeroUpperBitsForZExt(ZExt, CI, nullptr))
          return CI;
        return nullptr;
	    }

	    if (isa<Argument>(V))
      return nullptr;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return nullptr;

	    if (auto *PrevZExt = dyn_cast<ZExtInst>(I)) {
	      if (PrevZExt->getType() == WideTy)
	        return PrevZExt;

        return BuildDEFReplacementValue(ZExt, PrevZExt->getOperand(0), State);
	    }

	    if (auto *TI = dyn_cast<TruncInst>(I)) {
        if (hasZeroUpperBitsForZExt(ZExt, TI->getOperand(0), TI))
          return TI->getOperand(0);

	      return BuildDEFReplacementValue(ZExt, TI->getOperand(0), State);
	    }

    if (auto *SExt = dyn_cast<SExtInst>(I)) {
      if (SExt->getType() == WideTy && SExt->getSrcTy() == RootTy &&
          hasZeroUpperBitsForZExt(ZExt, SExt, SExt))
        return SExt;

      return BuildDEFReplacementValue(ZExt, SExt->getOperand(0), State);
    }

	    if (auto *PN = dyn_cast<PHINode>(I)) {
        if (PN->getType() == WideTy && hasZeroUpperBitsForZExt(ZExt, PN, PN))
          return PN;

	      PHINode *WidePhi =
	          PHINode::Create(WideTy, PN->getNumIncomingValues(),
                          PN->getName() + ".kz.wide", PN);
      State.Cache[PN] = WidePhi;

      for (unsigned K = 0; K < PN->getNumIncomingValues(); ++K) {
        Value *Incoming =
            BuildDEFReplacementValue(ZExt, PN->getIncomingValue(K), State);
        if (!Incoming) {
          State.Cache.erase(PN);
          WidePhi->eraseFromParent();
          return nullptr;
        }
        WidePhi->addIncoming(Incoming, PN->getIncomingBlock(K));
      }

      return WidePhi;
    }

	    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        if (BO->getType() != WideTy || !isDEFChainOpcode(BO->getOpcode()))
          return nullptr;

        if (hasZeroUpperBitsForZExt(ZExt, BO, BO))
          return BO;

	      Value *L = BuildDEFReplacementValue(ZExt, BO->getOperand(0), State);
	      Value *R = BuildDEFReplacementValue(ZExt, BO->getOperand(1), State);
      if (!L || !R)
        return nullptr;

      IRBuilder<> B(ZExt);
      Value *Wide =
          B.CreateBinOp(BO->getOpcode(), L, R, BO->getName() + ".kz.wide");
      State.Cache[BO] = Wide;
      return Wide;
    }

    return nullptr;
  }

  bool EliminateByDEF(ZExtInst *ZExt) {
    BuildState State;
    Value *Replacement = BuildDEFReplacementValue(ZExt, ZExt->getOperand(0), State);
    if (!Replacement || Replacement->getType() != ZExt->getType())
      return false;

    ZExt->replaceAllUsesWith(Replacement);
    ZExt->eraseFromParent();
    return true;
  }
*/


	  bool EliminateOneExtend(ZExtInst *ZExt) {
	    AnalyzeState State;
	    bool Required = false;

    for (User *U : ZExt->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI) {
        Required = true;
        break;
      }

      Required = AnalyzeUSE(ZExt, UI, State);
      if (Required)
        break;
    }

    if (!Required) {
      if (State.SawGEP && State.SawTruncSink) {
        errs() << "zext kept due to GEP + trunc sink path: " << *ZExt << "\n";
        return false;
      }

      errs() << "zext can be eliminated by USE: " << *ZExt << "\n";
      return ConvertZExtToSExt(ZExt);
    }

   /* Required = AnalyzeDEFValue(ZExt, ZExt->getOperand(0), State);

	    if (!Required) {
	      errs() << "zext can be eliminated by DEF: " << *ZExt << "\n";
	      return EliminateByDEF(ZExt);
	    }
*/
/*
      if (TryConvertZExtToSExt(ZExt)) {
        errs() << "zext can be converted to sext by KnownBits\n";
        return true;
      } */

	    errs() << "zext is still required: " << *ZExt << "\n";
	    return false;
	  }



  void print_logs(
      const std::vector<std::pair<BasicBlock*, uint64_t>> &hotness, std::vector<ZExtInst*> zexts) {

      errs() << "\n=== Function: " << F.getName() << " ===\n";

      for (auto &[BB, freq] : hotness) {
        errs() << "BasicBlock: ";
        BB->printAsOperand(errs(), false);
        errs() << " hotness=" << freq << "\n";
      }

      for (ZExtInst *I : zexts){
          errs() << *I << "\n";
      }
  }
};

} // namespace

PreservedAnalyses KawahitoZextAlgorithmPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  
  BlockFrequencyInfo &BFI = AM.getResult<BlockFrequencyAnalysis>(F);                                               
  EliminateRedundantZext Pass(F, BFI);
  bool Changed = Pass.run();

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

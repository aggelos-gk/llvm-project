//===-- AddZextAfterAlloca.cpp - Simple transitive type promotion ---------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Scalar/AddZextAfterAlloca.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>

using namespace llvm;

namespace {

class TypePromotion {
public:
  explicit TypePromotion(Function &F) : F(F) {}

  bool run() {
    if (F.isDeclaration())
      return false;

    buildPromotionPlan();
    bool Changed = promoteTypes();
    Changed |= prepareSafeExts();
    Changed |= rewriteRejectedExtUsers();
    Changed |= rewriteCrossBlockOriginalExtUsers();
    Changed |= rewriteUsers();
    Changed |= cleanupDeadInstructions();
    return Changed;
  }

private:
  Function &F;
  DenseMap<Value *, Value *> WideMap;
  DenseMap<Value *, Value *> TruncMap;
  DenseMap<Value *, Instruction *> InsertAnchorMap;
  DenseMap<Instruction *, Value *> SafeExtInputMap;
  DenseMap<Instruction *, Value *> RejectedExtInputMap;
  SmallPtrSet<Value *, 32> AllowedPromotionValues;
  SmallPtrSet<Instruction *, 16> SafeExts;
  SmallPtrSet<Instruction *, 16> RejectedExts;
  SmallPtrSet<Instruction *, 32> InternalInsts;
  SmallVector<Value *, 16> PromotedValues;
  SmallVector<WeakTrackingVH, 32> DeadInsts;

  Type *getI64Type() const { return Type::getInt64Ty(F.getContext()); }

  template <typename T>
  static auto copySameSignIfSupportedImpl(T *Dst, const T *Src, int)
      -> decltype(Dst->setSameSign(Src->hasSameSign()), void()) {
    Dst->setSameSign(Src->hasSameSign());
  }

  template <typename T>
  static void copySameSignIfSupportedImpl(T *, const T *, long) {}

  static void copySameSignIfSupported(ICmpInst *Dst, const ICmpInst *Src) {
    copySameSignIfSupportedImpl(Dst, Src, 0);
  }

  static bool isAnalysisInteger(Type *Ty) {
    auto *IntTy = dyn_cast_or_null<IntegerType>(Ty);
    return IntTy && IntTy->getBitWidth() < 64;
  }

  static bool isPromotableNarrowInteger(Type *Ty) {
    auto *IntTy = dyn_cast_or_null<IntegerType>(Ty);
    if (!IntTy)
      return false;

    unsigned BitWidth = IntTy->getBitWidth();
    return BitWidth > 1 && BitWidth < 64;
  }

  static bool isSafePathInteger(Type *Ty) {
    auto *IntTy = dyn_cast_or_null<IntegerType>(Ty);
    return IntTy && IntTy->getBitWidth() <= 64;
  }

  static bool isSourceRootValue(Value *V) {
    if (auto *Arg = dyn_cast<Argument>(V))
      return isAnalysisInteger(Arg->getType());

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;

    return (isa<LoadInst>(I) || isa<CallBase>(I)) && isAnalysisInteger(I->getType());
  }

  static bool isRelevantExtendInstruction(Instruction *I) {
    auto *Cast = dyn_cast_or_null<CastInst>(I);
    if (!Cast || (!isa<ZExtInst>(Cast) && !isa<SExtInst>(Cast)))
      return false;

    auto *SrcTy = dyn_cast<IntegerType>(Cast->getSrcTy());
    auto *DstTy = dyn_cast<IntegerType>(Cast->getDestTy());
    if (!SrcTy || !DstTy)
      return false;

    return SrcTy->getBitWidth() < DstTy->getBitWidth() &&
           DstTy->getBitWidth() <= 64 && SrcTy->getBitWidth() < 64;
  }

  static bool isPromotableOpcode(unsigned Opcode) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Mul:
    case Instruction::Sub:
    case Instruction::SDiv:
    case Instruction::SRem:
    case Instruction::Shl:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      return true;
    default:
      return false;
    }
  }

  static bool hasDisallowedNoWrapFlags(const BinaryOperator *BO) {
    switch (BO->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Shl:
      return BO->hasNoSignedWrap() || BO->hasNoUnsignedWrap();
    default:
      return false;
    }
  }

  static bool isSafePathBinaryInstruction(Instruction *I) {
    auto *BO = dyn_cast_or_null<BinaryOperator>(I);
    return BO && isSafePathInteger(BO->getType()) &&
           isPromotableOpcode(BO->getOpcode()) &&
           !hasDisallowedNoWrapFlags(BO);
  }

  static bool isPromotableBinaryInstruction(Instruction *I) {
    auto *BO = dyn_cast_or_null<BinaryOperator>(I);
    return BO && isPromotableNarrowInteger(BO->getType()) &&
           isPromotableOpcode(BO->getOpcode()) &&
           !hasDisallowedNoWrapFlags(BO);
  }

  static bool isSafePathPhiInstruction(Instruction *I) {
    auto *PN = dyn_cast_or_null<PHINode>(I);
    return PN && isSafePathInteger(PN->getType());
  }

  static bool isPromotablePhiInstruction(Instruction *I) {
    auto *PN = dyn_cast_or_null<PHINode>(I);
    return PN && isPromotableNarrowInteger(PN->getType());
  }

  static bool isPromotableICmpPredicate(CmpInst::Predicate Pred) {
    switch (Pred) {
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_ULT:
    case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SGT:
    case CmpInst::ICMP_SGE:
    case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_SLE:
      return true;
    default:
      return false;
    }
  }

  static bool isSafePathICmpInstruction(Instruction *I) {
    auto *Cmp = dyn_cast_or_null<ICmpInst>(I);
    return Cmp && isPromotableICmpPredicate(Cmp->getPredicate()) &&
           isSafePathInteger(Cmp->getOperand(0)->getType()) &&
           isSafePathInteger(Cmp->getOperand(1)->getType());
  }

  static bool isPromotableICmpInstruction(Instruction *I) {
    auto *Cmp = dyn_cast_or_null<ICmpInst>(I);
    return Cmp && isPromotableICmpPredicate(Cmp->getPredicate()) &&
           isPromotableNarrowInteger(Cmp->getOperand(0)->getType()) &&
           isPromotableNarrowInteger(Cmp->getOperand(1)->getType());
  }

  static bool isSafePathInstruction(Instruction *I) {
    return isSafePathBinaryInstruction(I) || isSafePathPhiInstruction(I) ||
           isSafePathICmpInstruction(I);
  }

  static bool isSinkInstruction(const Instruction *I) {
    return isa<StoreInst>(I) || isa<ReturnInst>(I) || isa<CallBase>(I) ||
           isa<GetElementPtrInst>(I);
  }

  void markInternal(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      InternalInsts.insert(I);
  }

  void rememberDead(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      DeadInsts.push_back(WeakTrackingVH(I));
  }

  void trackCreated(Value *V) {
    markInternal(V);
    rememberDead(V);
  }

  Instruction *getInsertPoint(Value *V) const {
    if (auto *Arg = dyn_cast<Argument>(V))
      return &*Arg->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return nullptr;

    if (auto *PN = dyn_cast<PHINode>(I))
      return PN->getParent()->getFirstNonPHI();

    if (I->isTerminator())
      return nullptr;

    return I->getNextNode();
  }

  Instruction *getSourceInsertPoint(Value *V) const {
    if (Instruction *Anchor = InsertAnchorMap.lookup(V))
      return Anchor;
    return getInsertPoint(V);
  }

  Value *getWideConstant(ConstantInt *CI) const {
    if (!isPromotableNarrowInteger(CI->getType()))
      return nullptr;

    return ConstantInt::get(getI64Type(), CI->getValue().sext(64));
  }

  bool isUpstreamPromotableImpl(Value *V, DenseMap<Value *, bool> &Memo,
                                SmallPtrSetImpl<Value *> &Visiting) const {
    auto It = Memo.find(V);
    if (It != Memo.end())
      return It->second;

    if (!Visiting.insert(V).second)
      return true;

    auto Finish = [&](bool Result) {
      Visiting.erase(V);
      Memo[V] = Result;
      return Result;
    };

    if (auto *CI = dyn_cast<ConstantInt>(V))
      return Finish(isAnalysisInteger(CI->getType()));

    if (isSourceRootValue(V))
      return Finish(true);

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return Finish(false);

    if (isPromotableBinaryInstruction(I))
      return Finish(isUpstreamPromotableImpl(I->getOperand(0), Memo, Visiting) &&
                    isUpstreamPromotableImpl(I->getOperand(1), Memo, Visiting));

    if (isPromotablePhiInstruction(I)) {
      auto *PN = cast<PHINode>(I);
      for (unsigned Idx = 0; Idx < PN->getNumIncomingValues(); ++Idx) {
        if (!isUpstreamPromotableImpl(PN->getIncomingValue(Idx), Memo, Visiting))
          return Finish(false);
      }
      return Finish(true);
    }

    if (isPromotableICmpInstruction(I))
      return Finish(isUpstreamPromotableImpl(I->getOperand(0), Memo, Visiting) &&
                    isUpstreamPromotableImpl(I->getOperand(1), Memo, Visiting));

    if (isRelevantExtendInstruction(I))
      return Finish(isUpstreamPromotableImpl(I->getOperand(0), Memo, Visiting));

    return Finish(false);
  }

  bool isUpstreamPromotable(Value *V, DenseMap<Value *, bool> &Memo) const {
    SmallPtrSet<Value *, 16> Visiting;
    return isUpstreamPromotableImpl(V, Memo, Visiting);
  }

  bool hasSafeRegularUpstreamImpl(Value *V, DenseMap<Value *, bool> &Memo,
                                  SmallPtrSetImpl<Value *> &Visiting) const {
    auto It = Memo.find(V);
    if (It != Memo.end())
      return It->second;

    if (!Visiting.insert(V).second)
      return false;

    auto Finish = [&](bool Result) {
      Visiting.erase(V);
      Memo[V] = Result;
      return Result;
    };

    if (isa<ConstantInt>(V) || isSourceRootValue(V))
      return Finish(false);

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return Finish(false);

    if (isSafePathInstruction(I))
      return Finish(true);

    if (isRelevantExtendInstruction(I))
      return Finish(hasSafeRegularUpstreamImpl(I->getOperand(0), Memo, Visiting));

    return Finish(false);
  }

  bool hasSafeRegularUpstream(Value *V, DenseMap<Value *, bool> &Memo) const {
    SmallPtrSet<Value *, 16> Visiting;
    return hasSafeRegularUpstreamImpl(V, Memo, Visiting);
  }

  bool hasSafeRegularDownstreamImpl(Value *V, DenseMap<Value *, bool> &UpstreamMemo,
                                    DenseMap<Value *, bool> &Memo,
                                    SmallPtrSetImpl<Value *> &Visiting) const {
    auto It = Memo.find(V);
    if (It != Memo.end())
      return It->second;

    if (!Visiting.insert(V).second)
      return false;

    auto Finish = [&](bool Result) {
      Visiting.erase(V);
      Memo[V] = Result;
      return Result;
    };

    auto *I = dyn_cast<Instruction>(V);
    if (I && isSafePathInstruction(I))
      return Finish(true);

    for (User *U : V->users()) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI)
        continue;

      if (isRelevantExtendInstruction(UserI)) {
        if (!isUpstreamPromotable(UserI, UpstreamMemo))
          continue;

        if (hasSafeRegularDownstreamImpl(UserI, UpstreamMemo, Memo, Visiting))
          return Finish(true);

        continue;
      }

      if (!isSafePathInstruction(UserI))
        continue;

      if (isPromotableBinaryInstruction(UserI) || isPromotablePhiInstruction(UserI) ||
          isPromotableICmpInstruction(UserI)) {
        if (isUpstreamPromotable(UserI, UpstreamMemo))
          return Finish(true);
        continue;
      }

      return Finish(true);
    }

    return Finish(false);
  }

  bool hasSafeRegularDownstream(Value *V, DenseMap<Value *, bool> &UpstreamMemo,
                                DenseMap<Value *, bool> &Memo) const {
    SmallPtrSet<Value *, 16> Visiting;
    return hasSafeRegularDownstreamImpl(V, UpstreamMemo, Memo, Visiting);
  }

  bool hasRealInstructionUpstream(Value *V) const {
    if (isa<ConstantInt>(V) || isSourceRootValue(V))
      return false;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;

    if (isRelevantExtendInstruction(I))
      return hasRealInstructionUpstream(I->getOperand(0));

    return true;
  }

  bool hasRealInstructionDownstreamImpl(Value *V, DenseMap<Value *, bool> &Memo,
                                        SmallPtrSetImpl<Value *> &Visiting) const {
    auto It = Memo.find(V);
    if (It != Memo.end())
      return It->second;

    if (!Visiting.insert(V).second)
      return false;

    auto Finish = [&](bool Result) {
      Visiting.erase(V);
      Memo[V] = Result;
      return Result;
    };

    for (User *U : V->users()) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI)
        continue;

      if (isRelevantExtendInstruction(UserI)) {
        if (hasRealInstructionDownstreamImpl(UserI, Memo, Visiting))
          return Finish(true);
        continue;
      }

      if (isSinkInstruction(UserI))
        continue;

      return Finish(true);
    }

    return Finish(false);
  }

  bool hasRealInstructionDownstream(Value *V, DenseMap<Value *, bool> &Memo) const {
    SmallPtrSet<Value *, 16> Visiting;
    return hasRealInstructionDownstreamImpl(V, Memo, Visiting);
  }

  bool hasUnsafeRealDownstreamImpl(Value *V, DenseMap<Value *, bool> &Memo,
                                   SmallPtrSetImpl<Value *> &Visiting) const {
    auto It = Memo.find(V);
    if (It != Memo.end())
      return It->second;

    if (!Visiting.insert(V).second)
      return false;

    auto Finish = [&](bool Result) {
      Visiting.erase(V);
      Memo[V] = Result;
      return Result;
    };

    for (User *U : V->users()) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI || isSinkInstruction(UserI))
        continue;

      if (isRelevantExtendInstruction(UserI)) {
        if (hasUnsafeRealDownstreamImpl(UserI, Memo, Visiting))
          return Finish(true);
        continue;
      }

      if (!isSafePathInstruction(UserI))
        return Finish(true);

      if (hasUnsafeRealDownstreamImpl(UserI, Memo, Visiting))
        return Finish(true);
    }

    return Finish(false);
  }

  bool hasUnsafeRealDownstream(Value *V, DenseMap<Value *, bool> &Memo) const {
    SmallPtrSet<Value *, 16> Visiting;
    return hasUnsafeRealDownstreamImpl(V, Memo, Visiting);
  }

  void markSafeBackward(Value *V, SmallPtrSetImpl<Value *> &Seen) {
    if (!Seen.insert(V).second)
      return;

    if (isa<ConstantInt>(V))
      return;

    if (auto *Arg = dyn_cast<Argument>(V)) {
      if (isPromotableNarrowInteger(Arg->getType()))
        AllowedPromotionValues.insert(Arg);
      return;
    }

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return;

    if (isSourceRootValue(I)) {
      if (isPromotableNarrowInteger(I->getType()))
        AllowedPromotionValues.insert(I);
      return;
    }

    if (auto *Ext = dyn_cast<CastInst>(I)) {
      if (!SafeExts.contains(Ext))
        return;
      markSafeBackward(Ext->getOperand(0), Seen);
      return;
    }

    if (isPromotableBinaryInstruction(I)) {
      AllowedPromotionValues.insert(I);
      markSafeBackward(I->getOperand(0), Seen);
      markSafeBackward(I->getOperand(1), Seen);
      return;
    }

    if (isPromotablePhiInstruction(I)) {
      AllowedPromotionValues.insert(I);
      auto *PN = cast<PHINode>(I);
      for (unsigned Idx = 0; Idx < PN->getNumIncomingValues(); ++Idx)
        markSafeBackward(PN->getIncomingValue(Idx), Seen);
      return;
    }

    if (isPromotableICmpInstruction(I)) {
      AllowedPromotionValues.insert(I);
      markSafeBackward(I->getOperand(0), Seen);
      markSafeBackward(I->getOperand(1), Seen);
    }
  }

  void markSafeForward(Value *V, SmallPtrSetImpl<Value *> &Seen) {
    if (!Seen.insert(V).second)
      return;

    for (User *U : V->users()) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI || isSinkInstruction(UserI))
        continue;

      if (auto *Ext = dyn_cast<CastInst>(UserI)) {
        if (!SafeExts.contains(Ext))
          continue;
        markSafeForward(Ext, Seen);
        continue;
      }

      if (!isSafePathInstruction(UserI))
        continue;

      if (isPromotableBinaryInstruction(UserI) || isPromotablePhiInstruction(UserI) ||
          isPromotableICmpInstruction(UserI))
        AllowedPromotionValues.insert(UserI);

      markSafeForward(UserI, Seen);
    }
  }

  void buildPromotionPlan() {
    AllowedPromotionValues.clear();
    SafeExts.clear();
    RejectedExts.clear();

    DenseMap<Value *, bool> UpstreamMemo;
    DenseMap<Value *, bool> UpstreamSafeMemo;
    DenseMap<Value *, bool> DownstreamMemo;
    DenseMap<Value *, bool> RealDownstreamMemo;
    DenseMap<Value *, bool> UnsafeDownstreamMemo;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isRelevantExtendInstruction(&I)) {
          bool Promotable = isUpstreamPromotable(&I, UpstreamMemo);
          bool SafeUpstream =
              hasSafeRegularUpstream(I.getOperand(0), UpstreamSafeMemo);
          bool SafeDownstream =
              hasSafeRegularDownstream(&I, UpstreamMemo, DownstreamMemo);
          bool UnsafeDownstream =
              hasUnsafeRealDownstream(&I, UnsafeDownstreamMemo);
          bool HasRealContext =
              hasRealInstructionUpstream(I.getOperand(0)) ||
              hasRealInstructionDownstream(&I, RealDownstreamMemo);

          if (Promotable && !UnsafeDownstream && (SafeUpstream || SafeDownstream))
            SafeExts.insert(&I);
          else if (HasRealContext)
            RejectedExts.insert(&I);

          continue;
        }
      }
    }

    SmallPtrSet<Value *, 32> BackwardSeen;
    SmallPtrSet<Value *, 32> ForwardSeen;
    for (Instruction *ExtI : SafeExts) {
      auto *Ext = cast<CastInst>(ExtI);
      markSafeBackward(Ext->getOperand(0), BackwardSeen);
      markSafeForward(Ext, ForwardSeen);
    }
  }

  Value *createSeedPromotion(Value *Original, Instruction *InsertBefore) {
    if (!InsertBefore || !isPromotableNarrowInteger(Original->getType()))
      return nullptr;

    if (Value *Wide = WideMap.lookup(Original))
      return Wide;

    StringRef BaseName = Original->hasName() ? Original->getName() : "tp";
    IRBuilder<> Builder(InsertBefore);
    Value *WideSeed =
        Builder.CreateSExt(Original, getI64Type(), BaseName + ".wide");
    Value *Trunc =
        Builder.CreateTrunc(WideSeed, Original->getType(), BaseName + ".trunc");
    Value *Wide =
        Builder.CreateSExt(Trunc, getI64Type(), BaseName + ".sext");

    WideMap[Original] = Wide;
    TruncMap[Original] = Trunc;
    InsertAnchorMap[Original] = InsertBefore;
    PromotedValues.push_back(Original);

    trackCreated(WideSeed);
    trackCreated(Trunc);
    trackCreated(Wide);
    return Wide;
  }

  Value *promoteBinary(BinaryOperator *BO) {
    if (Value *Wide = WideMap.lookup(BO))
      return Wide;

    Value *LHS = getWideValue(BO->getOperand(0));
    Value *RHS = getWideValue(BO->getOperand(1));
    if (!LHS || !RHS)
      return nullptr;

    StringRef BaseName = BO->hasName() ? BO->getName() : "tp";
    IRBuilder<> Builder(BO);
    Value *WideOp =
        Builder.CreateBinOp(BO->getOpcode(), LHS, RHS, BaseName + ".wide");
    Value *Trunc =
        Builder.CreateTrunc(WideOp, BO->getType(), BaseName + ".trunc");
    Value *Wide =
        Builder.CreateSExt(Trunc, getI64Type(), BaseName + ".sext");

    WideMap[BO] = Wide;
    TruncMap[BO] = Trunc;
    InsertAnchorMap[BO] = BO;
    PromotedValues.push_back(BO);

    trackCreated(WideOp);
    trackCreated(Trunc);
    trackCreated(Wide);
    rememberDead(BO);
    return Wide;
  }

  Value *getPhiInput(Value *V, BasicBlock *Pred) {
    if (Value *Wide = getWideValue(V))
      return Wide;

    if (auto *CI = dyn_cast<ConstantInt>(V))
      return getWideConstant(CI);

    auto *I = dyn_cast<Instruction>(V);
    if (I && I->getParent() == Pred && I->isTerminator())
      return nullptr;

    return nullptr;
  }

  Value *promotePhi(PHINode *PN) {
    if (Value *Wide = WideMap.lookup(PN))
      return Wide;

    StringRef BaseName = PN->hasName() ? PN->getName() : "phi";
    auto *WidePhi =
        PHINode::Create(getI64Type(), PN->getNumIncomingValues(),
                        BaseName + ".wide", PN);

    Instruction *InsertBefore = PN->getParent()->getFirstNonPHI();
    IRBuilder<> Builder(InsertBefore);
    Value *Trunc =
        Builder.CreateTrunc(WidePhi, PN->getType(), BaseName + ".trunc");
    Value *Wide =
        Builder.CreateSExt(Trunc, getI64Type(), BaseName + ".sext");

    WideMap[PN] = Wide;
    TruncMap[PN] = Trunc;
    InsertAnchorMap[PN] = InsertBefore;

    auto CleanupOnFailure = [&]() -> Value * {
      WideMap.erase(PN);
      TruncMap.erase(PN);
      InsertAnchorMap.erase(PN);
      Wide->replaceAllUsesWith(PoisonValue::get(Wide->getType()));
      Trunc->replaceAllUsesWith(PoisonValue::get(Trunc->getType()));
      cast<Instruction>(Wide)->eraseFromParent();
      cast<Instruction>(Trunc)->eraseFromParent();
      WidePhi->eraseFromParent();
      return nullptr;
    };

    SmallVector<std::pair<Value *, BasicBlock *>, 4> IncomingValues;
    for (unsigned I = 0; I < PN->getNumIncomingValues(); ++I) {
      BasicBlock *Pred = PN->getIncomingBlock(I);
      Value *Incoming = PN->getIncomingValue(I);
      Value *WideIncoming = getPhiInput(Incoming, Pred);
      if (!WideIncoming)
        return CleanupOnFailure();

      IncomingValues.emplace_back(WideIncoming, Pred);
    }

    for (auto [WideIncoming, Pred] : IncomingValues)
      WidePhi->addIncoming(WideIncoming, Pred);

    PromotedValues.push_back(PN);

    trackCreated(WidePhi);
    trackCreated(Trunc);
    trackCreated(Wide);
    rememberDead(PN);
    return Wide;
  }

  Value *getSourceShadow(Value *Original) {
    if (Value *Shadow = TruncMap.lookup(Original))
      return Shadow;
    return Original;
  }

  Value *prepareSafeExt(CastInst *Ext) {
    if (Value *Wide = WideMap.lookup(Ext))
      return Wide;

    Value *Original = Ext->getOperand(0);
    if (AllowedPromotionValues.contains(Original))
      (void)getWideValue(Original);

    Instruction *InsertBefore = getSourceInsertPoint(Original);
    if (!InsertBefore)
      return nullptr;

    Value *SourceInput = getSourceShadow(Original);
    StringRef BaseName = Original->hasName() ? Original->getName() : "tp";
    IRBuilder<> Builder(InsertBefore);
    Value *LocalWide = isa<ZExtInst>(Ext)
                           ? Builder.CreateSExt(SourceInput, getI64Type(),
                                                BaseName + ".zext.local")
                           : Builder.CreateSExt(SourceInput, getI64Type(),
                                                BaseName + ".sext.local");
    Value *LocalTrunc = Builder.CreateTrunc(
        LocalWide, Original->getType(),
        isa<ZExtInst>(Ext) ? BaseName + ".zext.trunc"
                           : BaseName + ".sext.trunc");

    Value *FinalWide = nullptr;
    if (Ext->getType()->isIntegerTy(64)) {
      FinalWide = Ext;
    } else {
      StringRef ExtName = Ext->hasName() ? Ext->getName() : "ext";
      FinalWide = isa<ZExtInst>(Ext)
                      ? Builder.CreateZExt(LocalTrunc, getI64Type(),
                                           ExtName + ".wide")
                      : Builder.CreateSExt(LocalTrunc, getI64Type(),
                                           ExtName + ".wide");
      Value *ResultTrunc = Builder.CreateTrunc(
          FinalWide, Ext->getType(), ExtName + ".trunc");
      TruncMap[Ext] = ResultTrunc;
      trackCreated(ResultTrunc);
      PromotedValues.push_back(Ext);
      trackCreated(FinalWide);
      rememberDead(Ext);
    }

    if (Ext->getOperand(0) != LocalTrunc)
      Ext->setOperand(0, LocalTrunc);

    SafeExtInputMap[Ext] = LocalTrunc;
    WideMap[Ext] = FinalWide;
    InsertAnchorMap[Ext] = InsertBefore;

    trackCreated(LocalWide);
    trackCreated(LocalTrunc);
    return FinalWide;
  }

  Value *getWideValue(Value *V) {
    if (Value *Wide = WideMap.lookup(V))
      return Wide;

    if (auto *CI = dyn_cast<ConstantInt>(V))
      return getWideConstant(CI);

    if (auto *Ext = dyn_cast<CastInst>(V)) {
      if (SafeExts.contains(Ext))
        return prepareSafeExt(Ext);
    }

    if (!isPromotableNarrowInteger(V->getType()))
      return nullptr;

    if (!AllowedPromotionValues.contains(V))
      return nullptr;

    if (auto *I = dyn_cast<Instruction>(V)) {
      if (isPromotableBinaryInstruction(I))
        return promoteBinary(cast<BinaryOperator>(I));
      if (isPromotablePhiInstruction(I))
        return promotePhi(cast<PHINode>(I));
      if (isSourceRootValue(I))
        return createSeedPromotion(I, getInsertPoint(I));
      return nullptr;
    }

    if (isa<Argument>(V))
      return createSeedPromotion(
          V, &*F.getEntryBlock().getFirstNonPHIOrDbgOrAlloca());

    return nullptr;
  }

  bool promoteICmp(ICmpInst *Cmp) {
    Value *LHS = getWideValue(Cmp->getOperand(0));
    Value *RHS = getWideValue(Cmp->getOperand(1));
    if (!LHS || !RHS)
      return false;

    StringRef BaseName = Cmp->hasName() ? Cmp->getName() : "icmp";
    auto *WideCmp = new ICmpInst(Cmp, Cmp->getPredicate(), LHS, RHS,
                                 BaseName + ".wide");
    copySameSignIfSupported(WideCmp, Cmp);

    Cmp->replaceAllUsesWith(WideCmp);
    trackCreated(WideCmp);
    rememberDead(Cmp);
    return true;
  }

  bool promoteTypes() {
    bool Changed = false;

    for (Argument &Arg : F.args()) {
      if (!AllowedPromotionValues.contains(&Arg))
        continue;
      Changed |= getWideValue(&Arg) != nullptr;
    }

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isSourceRootValue(&I) && AllowedPromotionValues.contains(&I) &&
            isPromotableNarrowInteger(I.getType())) {
          Changed |= getWideValue(&I) != nullptr;
          continue;
        }

        if (isPromotableICmpInstruction(&I) &&
            AllowedPromotionValues.contains(&I)) {
          Changed |= promoteICmp(cast<ICmpInst>(&I));
          continue;
        }

        if ((isPromotableBinaryInstruction(&I) || isPromotablePhiInstruction(&I)) &&
            AllowedPromotionValues.contains(&I))
          Changed |= getWideValue(&I) != nullptr;
      }
    }

    return Changed;
  }

  bool prepareSafeExts() {
    bool Changed = false;

    for (Instruction *ExtI : SafeExts) {
      auto *Ext = dyn_cast_or_null<CastInst>(ExtI);
      if (!Ext || !Ext->getParent())
        continue;

      if (SafeExtInputMap.contains(Ext))
        continue;

      Changed |= prepareSafeExt(Ext) != nullptr;
    }

    return Changed;
  }

  Value *getRejectedExtInputValue(CastInst *Ext) {
    if (Value *Existing = RejectedExtInputMap.lookup(Ext))
      return Existing;

    Value *Original = Ext->getOperand(0);
    if (AllowedPromotionValues.contains(Original))
      (void)getWideValue(Original);

    Instruction *InsertBefore = getSourceInsertPoint(Original);
    if (!InsertBefore)
      return nullptr;

    Value *SourceInput = getSourceShadow(Original);
    StringRef BaseName = Original->hasName() ? Original->getName() : "tp";
    IRBuilder<> Builder(InsertBefore);
    Value *LocalWide = isa<ZExtInst>(Ext)
                           ? Builder.CreateSExt(SourceInput, Ext->getType(),
                                                BaseName + ".zext.local")
                           : Builder.CreateSExt(SourceInput, Ext->getType(),
                                                BaseName + ".sext.local");
    Value *LocalTrunc = Builder.CreateTrunc(
        LocalWide, Original->getType(),
        isa<ZExtInst>(Ext) ? BaseName + ".zext.trunc"
                           : BaseName + ".sext.trunc");

    RejectedExtInputMap[Ext] = LocalTrunc;
    trackCreated(LocalWide);
    trackCreated(LocalTrunc);
    return LocalTrunc;
  }

  bool rewriteRejectedExtUsers() {
    bool Changed = false;

    for (Instruction *ExtI : RejectedExts) {
      auto *Ext = dyn_cast_or_null<CastInst>(ExtI);
      if (!Ext || !Ext->getParent())
        continue;

      Value *NewInput = getRejectedExtInputValue(Ext);
      if (!NewInput || Ext->getOperand(0) == NewInput)
        continue;

      Ext->setOperand(0, NewInput);
      Changed = true;
    }

    return Changed;
  }

  bool rewriteCrossBlockOriginalExtUsers() {
    SmallVector<Instruction *, 16> OriginalExts;
    for (Instruction *Ext : SafeExts)
      OriginalExts.push_back(Ext);
    for (Instruction *Ext : RejectedExts)
      OriginalExts.push_back(Ext);

    if (OriginalExts.empty())
      return false;

    DominatorTree DT(F);
    DenseMap<const Instruction *, unsigned> InstOrder;
    unsigned Order = 0;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        InstOrder[&I] = Order++;

    bool Changed = false;
    for (Instruction *ExtI : OriginalExts) {
      auto *Ext = dyn_cast_or_null<CastInst>(ExtI);
      if (!Ext || !Ext->getParent())
        continue;

      SmallVector<Instruction *, 8> ExternalUsers;
      for (User *U : Ext->users()) {
        auto *UserI = dyn_cast<Instruction>(U);
        if (!UserI || InternalInsts.contains(UserI))
          continue;
        ExternalUsers.push_back(UserI);
      }

      if (ExternalUsers.empty())
        continue;

      // If the original ext still feeds a direct PHI, skip the downstream
      // dummy-move entirely. Otherwise we keep both original and helper alive.
      if (llvm::any_of(ExternalUsers,
                       [](Instruction *UserI) { return isa<PHINode>(UserI); }))
        continue;

      SmallVector<Instruction *, 8> NonPhiExternalUsers(ExternalUsers.begin(),
                                                        ExternalUsers.end());

      if (NonPhiExternalUsers.empty())
        continue;

      auto *FirstUser = *std::min_element(
          NonPhiExternalUsers.begin(), NonPhiExternalUsers.end(),
          [&InstOrder](Instruction *LHS, Instruction *RHS) {
            return InstOrder.lookup(LHS) < InstOrder.lookup(RHS);
          });
      if (!FirstUser || FirstUser->getParent() == Ext->getParent())
        continue;

      StringRef ExtName = Ext->hasName() ? Ext->getName() : "ext";
      auto CreateLateHelper = [&](Instruction *InsertBefore) -> Value * {
        IRBuilder<> Builder(InsertBefore);
        Value *LateTrunc = Builder.CreateTrunc(
            Ext, Ext->getSrcTy(), ExtName + ".bb.trunc");
        Value *LateHelper = isa<ZExtInst>(Ext)
                                ? Builder.CreateZExt(LateTrunc, Ext->getType(),
                                                     ExtName + ".bb.zext")
                                : Builder.CreateSExt(LateTrunc, Ext->getType(),
                                                     ExtName + ".bb.sext");
        trackCreated(LateTrunc);
        trackCreated(LateHelper);
        return LateHelper;
      };

      Instruction *InsertBefore = FirstUser;
      if (!InsertBefore)
        continue;

      Value *BlockHelper = nullptr;
      auto *BlockHelperI = static_cast<Instruction *>(nullptr);
      auto IsDominatedByHelper = [&](Instruction *UserI) {
        if (UserI->getParent() == InsertBefore->getParent())
          return UserI == InsertBefore || InsertBefore->comesBefore(UserI);
        return DT.dominates(InsertBefore->getParent(), UserI->getParent());
      };

      if (!llvm::all_of(NonPhiExternalUsers, IsDominatedByHelper))
        continue;

      bool ReplacedAny = false;
      for (Instruction *UserI : NonPhiExternalUsers) {
        if (!BlockHelper) {
          BlockHelper = CreateLateHelper(InsertBefore);
          BlockHelperI = cast<Instruction>(BlockHelper);
        }

        UserI->replaceUsesOfWith(Ext, BlockHelper);
        ReplacedAny = true;
      }

      if (!ReplacedAny && BlockHelperI)
        RecursivelyDeleteTriviallyDeadInstructions(BlockHelperI);

      if (!ReplacedAny)
        continue;

      Changed = true;
    }

    return Changed;
  }

  bool rewritePromotedUsers(Value *Original) {
    Value *Trunc = TruncMap.lookup(Original);
    if (!Trunc)
      return false;

    SmallVector<User *, 8> Users;
    for (User *U : Original->users())
      Users.push_back(U);

    bool Changed = false;
    for (User *U : Users) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI || InternalInsts.contains(UserI) ||
          isRelevantExtendInstruction(UserI))
        continue;

      UserI->replaceUsesOfWith(Original, Trunc);
      Changed = true;
    }

    return Changed;
  }

  bool rewriteUsers() {
    bool Changed = false;

    for (Value *V : PromotedValues)
      Changed |= rewritePromotedUsers(V);

    return Changed;
  }

  bool cleanupDeadInstructions() {
    bool Changed = false;

    for (auto It = DeadInsts.rbegin(); It != DeadInsts.rend(); ++It) {
      Value *V = *It;
      auto *I = dyn_cast_or_null<Instruction>(V);
      if (!I || !isInstructionTriviallyDead(I))
        continue;

      RecursivelyDeleteTriviallyDeadInstructions(I);
      Changed = true;
    }

    return Changed;
  }
};

} // namespace

PreservedAnalyses AddZextAfterAllocaPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  (void)AM;
  TypePromotion Pass(F);
  bool Changed = Pass.run();

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

//===-- AddZextAfterAlloca.cpp - Simple transitive type promotion ---------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Scalar/AddZextAfterAlloca.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {

class TypePromotion {
public:
  explicit TypePromotion(Function &F) : F(F) {}

  bool run() {
    if (F.isDeclaration())
      return false;

    bool Changed = promoteTypes();
    Changed |= rewriteUsers();
    Changed |= cleanupDeadInstructions();
    return Changed;
  }

private:
  Function &F;
  DenseMap<Value *, Value *> WideMap;
  DenseMap<Instruction *, Value *> TruncMap;
  SmallVector<Value *, 16> ExtendedDefs;
  SmallVector<Instruction *, 16> PromotedInsts;
  SmallVector<WeakTrackingVH, 32> DeadInsts;

  Type *getI64Type() const { return Type::getInt64Ty(F.getContext()); }

  static bool isPromotableNarrowInteger(Type *Ty) {
    auto *IntTy = dyn_cast_or_null<IntegerType>(Ty);
    if (!IntTy)
      return false;

    unsigned BitWidth = IntTy->getBitWidth();
    return BitWidth > 1 && BitWidth < 64;
  }

  static bool isPromotableOpcode(unsigned Opcode) {
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

  static bool isPromotableBinaryInstruction(Instruction *I) {
    auto *BO = dyn_cast_or_null<BinaryOperator>(I);
    return BO && isPromotableNarrowInteger(BO->getType()) &&
           isPromotableOpcode(BO->getOpcode());
  }

  static bool isPromotablePhiInstruction(Instruction *I) {
    auto *PN = dyn_cast_or_null<PHINode>(I);
    return PN && isPromotableNarrowInteger(PN->getType());
  }

  static bool isPromotableInstruction(Instruction *I) {
    return isPromotableBinaryInstruction(I) || isPromotablePhiInstruction(I);
  }

  static bool isMatchingI64SExt(Instruction *I, Value *Original) {
    auto *SExt = dyn_cast<SExtInst>(I);
    return SExt && SExt->getOperand(0) == Original &&
           SExt->getType()->isIntegerTy(64);
  }

  static bool isMatchingI64ZExt(Instruction *I, Value *Original) {
    auto *ZExt = dyn_cast<ZExtInst>(I);
    return ZExt && ZExt->getOperand(0) == Original &&
           ZExt->getType()->isIntegerTy(64);
  }

  void rememberDead(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      DeadInsts.push_back(WeakTrackingVH(I));
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

  Value *getWideConstant(ConstantInt *CI) const {
    if (!isPromotableNarrowInteger(CI->getType()))
      return nullptr;

    return ConstantInt::get(getI64Type(), CI->getValue().sext(64));
  }

  Value *createExtend(Value *Original, Instruction *InsertBefore,
                      bool SaveValue) {
    if (!InsertBefore)
      return nullptr;

    StringRef BaseName = Original->hasName() ? Original->getName() : "tp";
    IRBuilder<> Builder(InsertBefore);
    Value *Wide = Builder.CreateSExt(Original, getI64Type(), BaseName + ".sext");

    if (SaveValue) {
      WideMap[Original] = Wide;
      ExtendedDefs.push_back(Original);
    }

    rememberDead(Wide);
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
    PromotedInsts.push_back(BO);

    rememberDead(WideOp);
    rememberDead(Trunc);
    rememberDead(Wide);
    rememberDead(BO);
    return Wide;
  }

  Value *getPhiInput(Value *V, BasicBlock *Pred) {
    if (Value *Wide = getWideValue(V))
      return Wide;

    if (!isPromotableNarrowInteger(V->getType()))
      return nullptr;

    auto *I = dyn_cast<Instruction>(V);
    if (I && I->getParent() == Pred && I->isTerminator())
      return nullptr;

    return createExtend(V, Pred->getTerminator(), false);
  }

  Value *promotePhi(PHINode *PN) {
    if (Value *Wide = WideMap.lookup(PN))
      return Wide;

    StringRef BaseName = PN->hasName() ? PN->getName() : "phi";
    PHINode *WidePhi =
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

    SmallVector<std::pair<Value *, BasicBlock *>, 4> IncomingValues;
    for (unsigned I = 0; I < PN->getNumIncomingValues(); ++I) {
      BasicBlock *Pred = PN->getIncomingBlock(I);
      Value *Incoming = PN->getIncomingValue(I);
      Value *WideIncoming = getPhiInput(Incoming, Pred);
      if (!WideIncoming)
        return nullptr;

      IncomingValues.emplace_back(WideIncoming, Pred);
    }

    for (auto [WideIncoming, Pred] : IncomingValues)
      WidePhi->addIncoming(WideIncoming, Pred);

    PromotedInsts.push_back(PN);

    rememberDead(WidePhi);
    rememberDead(Trunc);
    rememberDead(Wide);
    rememberDead(PN);
    return Wide;
  }

  Value *getWideValue(Value *V) {
    if (Value *Wide = WideMap.lookup(V))
      return Wide;

    if (auto *CI = dyn_cast<ConstantInt>(V))
      return getWideConstant(CI);

    if (!isPromotableNarrowInteger(V->getType()))
      return nullptr;

    if (auto *I = dyn_cast<Instruction>(V)) {
      if (isPromotableBinaryInstruction(I))
        return promoteBinary(cast<BinaryOperator>(I));
      if (isPromotablePhiInstruction(I))
        return promotePhi(cast<PHINode>(I));

      return createExtend(I, getInsertPoint(I), true);
    }

    if (isa<Argument>(V))
      return createExtend(
          V, &*F.getEntryBlock().getFirstNonPHIOrDbgOrAlloca(), true);

    return nullptr;
  }

  bool promoteTypes() {
    bool Changed = false;

    for (Argument &Arg : F.args()) {
      if (!isPromotableNarrowInteger(Arg.getType()))
        continue;
      Changed |= getWideValue(&Arg) != nullptr;
    }

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (isPromotableNarrowInteger(LI->getType()))
            Changed |= getWideValue(LI) != nullptr;
          continue;
        }

        if (isPromotableInstruction(&I))
          Changed |= getWideValue(&I) != nullptr;
      }
    }

    return Changed;
  }

  bool rewriteExtendedUsers(Value *Original) {
    Value *Wide = WideMap.lookup(Original);
    if (!Wide)
      return false;

    SmallVector<User *, 8> Users;
    for (User *U : Original->users())
      Users.push_back(U);
    bool Changed = false;

    for (User *U : Users) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI || UserI == Wide)
        continue;

      if (!isMatchingI64SExt(UserI, Original))
        continue;

      UserI->replaceAllUsesWith(Wide);
      rememberDead(UserI);
      Changed = true;
    }

    return Changed;
  }

  bool rewritePromotedUsers(Instruction *Original) {
    Value *Trunc = TruncMap.lookup(Original);
    Value *Wide = WideMap.lookup(Original);
    if (!Trunc || !Wide)
      return false;

    SmallVector<User *, 8> Users;
    for (User *U : Original->users())
      Users.push_back(U);
    bool Changed = false;

    for (User *U : Users) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI || UserI == Wide || UserI == Trunc)
        continue;

      if (isMatchingI64ZExt(UserI, Original)) {
        UserI->replaceUsesOfWith(Original, Trunc);
        Changed = true;
        continue;
      }

      if (isMatchingI64SExt(UserI, Original)) {
        UserI->replaceAllUsesWith(Wide);
        rememberDead(UserI);
        Changed = true;
        continue;
      }

      UserI->replaceUsesOfWith(Original, Trunc);
      Changed = true;
    }

    return Changed;
  }

  bool rewriteUsers() {
    bool Changed = false;

    for (Instruction *I : PromotedInsts) {
      if (!I->getParent())
        continue;
      Changed |= rewritePromotedUsers(I);
    }

    for (Value *V : ExtendedDefs)
      Changed |= rewriteExtendedUsers(V);

    return Changed;
  }

  bool cleanupDeadInstructions() {
    bool Changed = false;

    for (auto It = DeadInsts.rbegin(); It != DeadInsts.rend();
         ++It) {
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

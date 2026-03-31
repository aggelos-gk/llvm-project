//===-- AddZextAfterAlloca.cpp - Kawahito-style promotion ---------------*- C++ -*-===//
//
// Kawahito Algorithm implementation:
// 1. Zext after EVERY definition point of small integers (i8, i16, i32) -> i64
//    Definition points: Load, PHI, Function Arguments, Call results, Trunc results
//    (NOT after arithmetic - those get promoted to i64 directly)
// 2. Promote arithmetic operations (add, sub, mul, and, or, xor, etc.) to i64
// 3. Truncate results back to original type only when used by non-arithmetic
//    operations or when type mismatch would occur (returns, stores, compares, etc.)
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/AddZextAfterAlloca.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace {

static bool isSmallIntType(Type *Ty) {
  return Ty->isIntegerTy(8) || Ty->isIntegerTy(16) || Ty->isIntegerTy(32);
}

static bool isPromotableBinOp(unsigned Opcode) {
  return Opcode == Instruction::Add || Opcode == Instruction::Sub ||
         Opcode == Instruction::Mul || Opcode == Instruction::And ||
         Opcode == Instruction::Or || Opcode == Instruction::Xor ||
         Opcode == Instruction::Shl || Opcode == Instruction::LShr ||
         Opcode == Instruction::AShr || Opcode == Instruction::UDiv ||
         Opcode == Instruction::SDiv || Opcode == Instruction::URem ||
         Opcode == Instruction::SRem;
}

static bool isOurZExt(Value *V) {
  if (ZExtInst *ZExt = dyn_cast<ZExtInst>(V))
    return ZExt->getName().ends_with(".zext");
  return false;
}

static bool isOurTrunc(Value *V) {
  if (TruncInst *Trunc = dyn_cast<TruncInst>(V))
    return Trunc->getName().ends_with(".trunc");
  return false;
}

static bool isI64Value(Value *V) {
  return V->getType()->isIntegerTy(64);
}

static bool isDefinitionPoint(Instruction *I) {
  if (isa<LoadInst>(I))    return true;
  if (isa<PHINode>(I))     return true;  // seeds BinOp promotion; Phase 2b upgrades PHIs to i64 directly
  if (isa<CallInst>(I))    return true;
  if (isa<TruncInst>(I))   return true;
  return false;
}

static bool userNeedsOriginalType(Value *Original, Use &U) {
  User *UserInst = U.getUser();
  Type *OrigType = Original->getType();

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(UserInst)) {
    if (isPromotableBinOp(BinOp->getOpcode()))
      return false;
  }

  if (isOurZExt(UserInst) || isOurTrunc(UserInst))
    return false;

  // FIX 1: A zext-to-i64 user does NOT need the original small type —
  // it wants an i64 result anyway.  We will replace the zext with the
  // promoted i64 value directly (the cleanup step below handles this, but
  // we must NOT redirect its operand to a trunc first or the cleanup loses
  // track of it).  Returning false here causes Phase 3 to put this use in
  // UsesToReplace, so the zext's SOURCE operand becomes the i64 Extended
  // value — which makes the zext redundant (zext i64 X to i64 is invalid
  // anyway, so LLVM will fold it; more importantly the cleanup step will
  // then see a zext whose source is already i64 and remove it).
  //
  // Concretely: given  %9 = zext i32 %8 to i64  and we promoted %8 -> %8_i64,
  // we want all *users of %9* to use %8_i64, then erase %9.
  // We achieve that by: (a) NOT putting this use in UsesNeedingTrunc, so the
  // zext's operand stays as %8_orig until the dedicated cleanup below replaces
  // the entire zext; and (b) collecting the zext for wholesale replacement.
  if (ZExtInst *ZExt = dyn_cast<ZExtInst>(UserInst)) {
    if (ZExt->getDestTy()->isIntegerTy(64))
      return false; // handled by the pre-existing-zext cleanup step
  }

  if (ReturnInst *Ret = dyn_cast<ReturnInst>(UserInst)) {
    Type *RetTy = Ret->getParent()->getParent()->getReturnType();
    return RetTy == OrigType;
  }

  if (StoreInst *Store = dyn_cast<StoreInst>(UserInst)) {
    if (Store->getValueOperand() == Original) {
      Type *DestTy = nullptr;
      if (AllocaInst *AI = dyn_cast<AllocaInst>(Store->getPointerOperand()))
        DestTy = AI->getAllocatedType();
      else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Store->getPointerOperand()))
        DestTy = GV->getValueType();
      if (DestTy && DestTy == OrigType)
        return true;
    }
  }

  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UserInst)) {
    Value *Other = (ICmp->getOperand(0) == Original) ?
                    ICmp->getOperand(1) : ICmp->getOperand(0);
    return Other->getType() == OrigType;
  }

  if (PHINode *PHI = dyn_cast<PHINode>(UserInst))
    return PHI->getType() == OrigType;

  if (SelectInst *Sel = dyn_cast<SelectInst>(UserInst))
    return Sel->getType() == OrigType;

  if (CallInst *Call = dyn_cast<CallInst>(UserInst)) {
    for (unsigned i = 0; i < Call->arg_size(); ++i) {
      if (Call->getArgOperand(i) == Original) {
        if (Function *F = Call->getCalledFunction()) {
          if (i < F->arg_size())
            return F->getArg(i)->getType() == OrigType;
        }
        return true;
      }
    }
  }

  if (isa<TruncInst>(UserInst) || isa<SExtInst>(UserInst) ||
      isa<ZExtInst>(UserInst) || isa<BitCastInst>(UserInst) ||
      isa<IntToPtrInst>(UserInst) || isa<PtrToIntInst>(UserInst))
    return true;

  if (isa<LandingPadInst>(UserInst) || isa<CatchPadInst>(UserInst))
    return true;

  return true;
}

static void insertZextAfter(Instruction *I, Type *I64Ty,
                             DenseMap<Value*, Value*> &ExtendedValues,
                             bool &LocalChanged) {
  if (!isSmallIntType(I->getType())) return;
  if (ExtendedValues.count(I)) return;
  if (isOurZExt(I) || isOurTrunc(I)) return;

  IRBuilder<> Builder(I->getNextNode());
  Value *ZExt = Builder.CreateZExt(I, I64Ty, I->getName() + ".zext");
  ExtendedValues[I] = ZExt;
  LocalChanged = true;
}

class AddZextAfterAlloca : public FunctionPass {
public:
  static char ID;
  AddZextAfterAlloca() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Changed = false;
    Type *I64Ty = Type::getInt64Ty(F.getContext());

    DenseMap<Value*, Value*> ExtendedValues;
    DenseSet<BinaryOperator*> PromotedBinOps;
    DenseSet<PHINode*> PromotedPHIs;

    for (Argument &Arg : F.args()) {
      if (!isSmallIntType(Arg.getType())) continue;
      if (ExtendedValues.count(&Arg)) continue;
      Instruction *InsertPt = &F.getEntryBlock().front();
      IRBuilder<> Builder(InsertPt);
      Value *ZExt = Builder.CreateZExt(&Arg, I64Ty, Arg.getName() + ".zext");
      ExtendedValues[&Arg] = ZExt;
      Changed = true;
    }

    bool LocalChanged;
    do {
      LocalChanged = false;

      // Phase 1: Zext after non-PHI definition points (Load, Call, Trunc).
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (!isSmallIntType(I.getType())) continue;
          if (isI64Value(&I)) continue;
          if (ExtendedValues.count(&I)) continue;
          if (!isDefinitionPoint(&I)) continue;

          insertZextAfter(&I, I64Ty, ExtendedValues, LocalChanged);
        }
      }

      // Phase 2: Promote BinOps to i64.
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&I);
          if (!BinOp) continue;
          if (!isPromotableBinOp(BinOp->getOpcode())) continue;
          if (PromotedBinOps.count(BinOp)) continue;
          if (!isSmallIntType(BinOp->getType())) continue;

          bool ShouldPromote = false;
          SmallVector<Value*, 2> NewOperands;

          for (Value *Op : BinOp->operands()) {
            if (isI64Value(Op)) {
              NewOperands.push_back(Op);
              ShouldPromote = true;
            } else if (ExtendedValues.count(Op)) {
              NewOperands.push_back(ExtendedValues[Op]);
              ShouldPromote = true;
            } else if (Instruction *OpI = dyn_cast<Instruction>(Op)) {
              if (isSmallIntType(OpI->getType()) && isDefinitionPoint(OpI)) {
                insertZextAfter(OpI, I64Ty, ExtendedValues, LocalChanged);
                NewOperands.push_back(ExtendedValues[OpI]);
                ShouldPromote = true;
              } else {
                NewOperands.push_back(Op);
              }
            } else if (Argument *ArgOp = dyn_cast<Argument>(Op)) {
              if (ExtendedValues.count(ArgOp)) {
                NewOperands.push_back(ExtendedValues[ArgOp]);
                ShouldPromote = true;
              } else {
                NewOperands.push_back(Op);
              }
            } else {
              NewOperands.push_back(Op);
            }
          }

          if (!ShouldPromote) continue;

          IRBuilder<> Builder(BinOp);
          for (Value *&Op : NewOperands) {
            if (!Op->getType()->isIntegerTy(64))
              Op = Builder.CreateZExt(Op, I64Ty, Op->getName() + ".zext");
          }

          Value *NewBinOp = nullptr;
          std::string OrigName = BinOp->getName().str();

          switch (BinOp->getOpcode()) {
            case Instruction::Add:  NewBinOp = Builder.CreateAdd(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::Sub:  NewBinOp = Builder.CreateSub(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::Mul:  NewBinOp = Builder.CreateMul(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::And:  NewBinOp = Builder.CreateAnd(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::Or:   NewBinOp = Builder.CreateOr(NewOperands[0], NewOperands[1], OrigName);  break;
            case Instruction::Xor:  NewBinOp = Builder.CreateXor(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::Shl:  NewBinOp = Builder.CreateShl(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::LShr: NewBinOp = Builder.CreateLShr(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::AShr: NewBinOp = Builder.CreateAShr(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::UDiv: NewBinOp = Builder.CreateUDiv(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::SDiv: NewBinOp = Builder.CreateSDiv(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::URem: NewBinOp = Builder.CreateURem(NewOperands[0], NewOperands[1], OrigName); break;
            case Instruction::SRem: NewBinOp = Builder.CreateSRem(NewOperands[0], NewOperands[1], OrigName); break;
            default: continue;
          }

          ExtendedValues[BinOp] = NewBinOp;
          PromotedBinOps.insert(BinOp);
          LocalChanged = true;
        }
      }

      // Phase 2b: Upgrade zext'd PHIs to full i64 PHIs.
      //
      // Phase 1 inserted "%.zext = zext i32 %phi to i64" after each small-int
      // PHI.  Here we go further: if we can build a proper i64 PHI (all
      // incoming values have i64 forms), we create one and replace the zext.
      // This removes the zext+trunc round-trip for back-edges entirely.
      //
      // We only upgrade a PHI when ALL incoming values have known i64 forms
      // so we don't create placeholder zexts in predecessor blocks.  The
      // outer do-loop re-runs until stable, so back-edge PHIs get upgraded
      // once their incoming promoted value is registered in ExtendedValues.
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          PHINode *Phi = dyn_cast<PHINode>(&I);
          if (!Phi) continue;
          if (!isSmallIntType(Phi->getType())) continue;
          if (PromotedPHIs.count(Phi)) continue; // already upgraded
          // Only upgrade PHIs that Phase 1 already zext'd (i.e. they are
          // in ExtendedValues with a ZExtInst as the extended value).
          if (!ExtendedValues.count(Phi)) continue;
          if (!isa<ZExtInst>(ExtendedValues[Phi])) continue;

          // Helper: best available i64 form for an incoming value.
          auto getI64Form = [&](Value *Inc) -> Value* {
            if (isI64Value(Inc)) return Inc;
            if (ExtendedValues.count(Inc)) return ExtendedValues[Inc];
            if (TruncInst *Tr = dyn_cast<TruncInst>(Inc))
              if (isOurTrunc(Tr) && isI64Value(Tr->getOperand(0)))
                return Tr->getOperand(0);
            // Small integer constants: produce the equivalent i64 constant.
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc))
              if (isSmallIntType(CI->getType()))
                return ConstantInt::get(I64Ty, CI->getZExtValue());
            return nullptr;
          };

          // Require ALL incoming values to have an i64 form before upgrading.
          bool AllKnown = true;
          for (Value *Inc : Phi->incoming_values())
            if (!getI64Form(Inc)) { AllKnown = false; break; }
          if (!AllKnown) continue;

          // Build the i64 PHI before the original.
          IRBuilder<> Builder(Phi);
          PHINode *NewPhi = Builder.CreatePHI(I64Ty,
                                              Phi->getNumIncomingValues(),
                                              Phi->getName());
          for (unsigned i = 0, e = Phi->getNumIncomingValues(); i < e; ++i)
            NewPhi->addIncoming(getI64Form(Phi->getIncomingValue(i)),
                                Phi->getIncomingBlock(i));

          // Replace the old zext entry with the new i64 PHI.
          Value *OldZExt = ExtendedValues[Phi];
          OldZExt->replaceAllUsesWith(NewPhi);
          if (Instruction *OldZExtI = dyn_cast<Instruction>(OldZExt))
            if (OldZExtI->use_empty())
              OldZExtI->eraseFromParent();

          // Clear all incoming values of the original PHI so that the
          // instructions feeding it (e.g. a promoted BinOp) lose this use
          // and become use_empty, allowing them to be erased later.
          // We replace each incoming with poison to satisfy LLVM's IR rules.
          Value *Poison = PoisonValue::get(Phi->getType());
          for (unsigned i = 0, e = Phi->getNumIncomingValues(); i < e; ++i)
            Phi->setIncomingValue(i, Poison);

          ExtendedValues[Phi] = NewPhi;
          PromotedPHIs.insert(Phi);
          LocalChanged = true;
        }
      }

      // Phase 2c: Fix up incoming values of already-promoted PHIs whose
      // back-edge values now have better (canonical) i64 forms available.
      for (PHINode *OldPhi : PromotedPHIs) {
        PHINode *NewPhi = cast<PHINode>(ExtendedValues[OldPhi]);
        for (unsigned i = 0, e = OldPhi->getNumIncomingValues(); i < e; ++i) {
          Value *OldInc = OldPhi->getIncomingValue(i);
          Value *CurInc = NewPhi->getIncomingValue(i);
          if (!isI64Value(CurInc)) continue;

          Value *Better = nullptr;
          if (ExtendedValues.count(OldInc)) {
            Value *Cand = ExtendedValues[OldInc];
            if (Cand->getType()->isIntegerTy(64) && Cand != CurInc)
              Better = Cand;
          }
          if (!Better) {
            if (TruncInst *Tr = dyn_cast<TruncInst>(OldInc))
              if (isOurTrunc(Tr) && isI64Value(Tr->getOperand(0)) &&
                  Tr->getOperand(0) != CurInc)
                Better = Tr->getOperand(0);
          }
          if (Better) {
            NewPhi->setIncomingValue(i, Better);
            LocalChanged = true;
          }
        }
      }

      Changed |= LocalChanged;
    } while (LocalChanged);

    // Pre-Phase-3: fold pre-existing zexts whose source is now promoted.
    // Must run BEFORE Phase 3 so that ExtendedValues still maps the original
    // small-int value; Phase 3 would otherwise redirect the zext's operand to
    // a trunc and break the lookup.
    for (BasicBlock &BB : F) {
      SmallVector<Instruction*, 8> ToErase;
      for (Instruction &I : BB) {
        ZExtInst *ZExt = dyn_cast<ZExtInst>(&I);
        if (!ZExt || !ZExt->getDestTy()->isIntegerTy(64)) continue;
        if (isOurZExt(ZExt)) continue;
        Value *ZExtSrc = ZExt->getOperand(0);
        if (ExtendedValues.count(ZExtSrc)) {
          Value *Promoted = ExtendedValues[ZExtSrc];
          if (Promoted->getType()->isIntegerTy(64) && Promoted != ZExt) {
            ZExt->replaceAllUsesWith(Promoted);
            ToErase.push_back(ZExt);
            Changed = true;
          }
        }
      }
      for (Instruction *Dead : ToErase)
        if (Dead->use_empty())
          Dead->eraseFromParent();
    }

    // Phase 3: Replace uses of original small-int values with their promoted
    // i64 counterparts, inserting truncs only where the original narrow type
    // is truly required (stores, returns, calls expecting small type, etc.).
    DenseMap<Value*, Value*> TruncCache;

    for (auto &Pair : ExtendedValues) {
      Value *Original = Pair.first;
      Value *Extended = Pair.second;
      Type *OrigType = Original->getType();

      if (!isSmallIntType(OrigType)) continue;

      SmallVector<Use*, 16> UsesToReplace;
      SmallVector<Use*, 16> UsesNeedingTrunc;

      for (Use &U : Original->uses()) {
        User *UserInst = U.getUser();
        if (UserInst == Extended) continue;
        // Skip users that will be erased (promoted BinOps and promoted PHIs).
        if (BinaryOperator *UBO = dyn_cast<BinaryOperator>(UserInst))
          if (PromotedBinOps.count(UBO)) continue;
        if (PHINode *UPHI = dyn_cast<PHINode>(UserInst))
          if (PromotedPHIs.count(UPHI)) continue;

        if (userNeedsOriginalType(Original, U))
          UsesNeedingTrunc.push_back(&U);
        else
          UsesToReplace.push_back(&U);
      }

      for (Use *U : UsesToReplace)
        U->set(Extended);

      if (!UsesNeedingTrunc.empty()) {
        Instruction *ExtInst = dyn_cast<Instruction>(Extended);
        if (!ExtInst) continue;

        Value *Trunc = nullptr;
        auto It = TruncCache.find(Extended);
        if (It != TruncCache.end()) {
          Trunc = It->second;
        } else {
          IRBuilder<> Builder(ExtInst->getNextNode());
          Trunc = Builder.CreateTrunc(Extended, OrigType,
                                      Original->getName() + ".trunc");
          TruncCache[Extended] = Trunc;
        }
        for (Use *U : UsesNeedingTrunc)
          U->set(Trunc);
      }
    }

    // Erase dead promoted BinOps and PHIs.
    // Use a worklist: RAUW(poison) on each candidate to drop all uses it has
    // from other promoted instructions, then erase once truly use_empty.
    // Iterate until stable so that erasing one instruction makes another dead.
    {
      SmallVector<Instruction*, 16> Worklist;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB) {
          if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I))
            if (PromotedBinOps.count(BO)) Worklist.push_back(BO);
          if (PHINode *PH = dyn_cast<PHINode>(&I))
            if (PromotedPHIs.count(PH)) Worklist.push_back(PH);
        }
      // RAUW with poison so all uses (including cross-references between
      // promoted instructions) are dropped, then erase.
      for (Instruction *I : Worklist) {
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
        I->eraseFromParent();
        Changed = true;
      }
    }

    return Changed;
  }
};

} // namespace

char AddZextAfterAlloca::ID = 0;

INITIALIZE_PASS(AddZextAfterAlloca, "add-zext-alloca",
                "Kawahito-style i64 promotion", false, false)

FunctionPass *llvm::createAddZextAfterAllocaPass() {
  return new AddZextAfterAlloca();
}

// ============================================================
// New Pass Manager implementation
// ============================================================
PreservedAnalyses AddZextAfterAllocaPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  bool Changed = false;
  Type *I64Ty = Type::getInt64Ty(F.getContext());

  auto isSmallIntType = [](Type *Ty) -> bool {
    return Ty->isIntegerTy(8) || Ty->isIntegerTy(16) || Ty->isIntegerTy(32);
  };

  auto isPromotableBinOp = [](unsigned Opcode) -> bool {
    return Opcode == Instruction::Add || Opcode == Instruction::Sub ||
           Opcode == Instruction::Mul || Opcode == Instruction::And ||
           Opcode == Instruction::Or || Opcode == Instruction::Xor ||
           Opcode == Instruction::Shl || Opcode == Instruction::LShr ||
           Opcode == Instruction::AShr || Opcode == Instruction::UDiv ||
           Opcode == Instruction::SDiv || Opcode == Instruction::URem ||
           Opcode == Instruction::SRem;
  };

  auto isOurZExt = [](Value *V) -> bool {
    if (ZExtInst *ZExt = dyn_cast<ZExtInst>(V))
      return ZExt->getName().ends_with(".zext");
    return false;
  };

  auto isOurTrunc = [](Value *V) -> bool {
    if (TruncInst *Trunc = dyn_cast<TruncInst>(V))
      return Trunc->getName().ends_with(".trunc");
    return false;
  };

  auto isI64Value = [](Value *V) -> bool {
    return V->getType()->isIntegerTy(64);
  };

  auto isDefinitionPoint = [](Instruction *I) -> bool {
    // PHIs remain definition points to seed BinOp promotion.
    // Phase 2b will additionally upgrade any zext'd PHI to a full i64 PHI.
    return isa<LoadInst>(I) || isa<PHINode>(I) ||
           isa<CallInst>(I) || isa<TruncInst>(I);
  };

  auto userNeedsOriginalType = [&](Value *Original, Use &U) -> bool {
    User *UserInst = U.getUser();
    Type *OrigType = Original->getType();

    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(UserInst)) {
      if (isPromotableBinOp(BinOp->getOpcode()))
        return false;
    }

    if (isOurZExt(UserInst) || isOurTrunc(UserInst))
      return false;

    // FIX 1: zext-to-i64 users are handled by the pre-existing-zext cleanup.
    // Do not redirect their operand to a trunc — let them keep pointing at
    // the original small-int value so the cleanup step can find and fold them.
    if (ZExtInst *ZExt = dyn_cast<ZExtInst>(UserInst)) {
      if (ZExt->getDestTy()->isIntegerTy(64))
        return false;
    }

    if (ReturnInst *Ret = dyn_cast<ReturnInst>(UserInst)) {
      Type *RetTy = Ret->getParent()->getParent()->getReturnType();
      return RetTy == OrigType;
    }

    if (StoreInst *Store = dyn_cast<StoreInst>(UserInst)) {
      if (Store->getValueOperand() == Original) {
        Type *DestTy = nullptr;
        if (AllocaInst *AI = dyn_cast<AllocaInst>(Store->getPointerOperand()))
          DestTy = AI->getAllocatedType();
        else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Store->getPointerOperand()))
          DestTy = GV->getValueType();
        return DestTy && DestTy == OrigType;
      }
    }

    if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UserInst)) {
      Value *Other = (ICmp->getOperand(0) == Original) ?
                      ICmp->getOperand(1) : ICmp->getOperand(0);
      return Other->getType() == OrigType;
    }

    if (PHINode *PHI = dyn_cast<PHINode>(UserInst))
      return PHI->getType() == OrigType;

    if (SelectInst *Sel = dyn_cast<SelectInst>(UserInst))
      return Sel->getType() == OrigType;

    if (CallInst *Call = dyn_cast<CallInst>(UserInst)) {
      for (unsigned i = 0; i < Call->arg_size(); ++i) {
        if (Call->getArgOperand(i) == Original) {
          if (Function *Func = Call->getCalledFunction()) {
            if (i < Func->arg_size())
              return Func->getArg(i)->getType() == OrigType;
          }
          return true;
        }
      }
    }

    if (isa<TruncInst>(UserInst) || isa<SExtInst>(UserInst) ||
        isa<ZExtInst>(UserInst) || isa<BitCastInst>(UserInst) ||
        isa<IntToPtrInst>(UserInst) || isa<PtrToIntInst>(UserInst))
      return true;

    if (isa<LandingPadInst>(UserInst) || isa<CatchPadInst>(UserInst))
      return true;

    return true;
  };

  DenseMap<Value*, Value*> ExtendedValues;
  DenseSet<BinaryOperator*> PromotedBinOps;
  DenseSet<PHINode*> PromotedPHIs;

  // Handle function arguments first
  for (Argument &Arg : F.args()) {
    if (!isSmallIntType(Arg.getType())) continue;
    if (ExtendedValues.count(&Arg)) continue;
    IRBuilder<> Builder(&F.getEntryBlock().front());
    Value *ZExt = Builder.CreateZExt(&Arg, I64Ty, Arg.getName() + ".zext");
    ExtendedValues[&Arg] = ZExt;
    Changed = true;
  }

  bool LocalChanged;
  do {
    LocalChanged = false;

    // Phase 1: Zext after non-PHI definition points (Load, Call, Trunc).
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (!isSmallIntType(I.getType())) continue;
        if (isI64Value(&I)) continue;
        if (ExtendedValues.count(&I)) continue;
        if (!isDefinitionPoint(&I)) continue;
        if (isOurZExt(&I) || isOurTrunc(&I)) continue;

        IRBuilder<> Builder(I.getNextNode());
        Value *ZExt = Builder.CreateZExt(&I, I64Ty, I.getName() + ".zext");
        ExtendedValues[&I] = ZExt;
        LocalChanged = true;
      }
    }

    // Phase 2: Promote BinOps to i64.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&I);
        if (!BinOp) continue;
        if (!isPromotableBinOp(BinOp->getOpcode())) continue;
        if (PromotedBinOps.count(BinOp)) continue;
        if (!isSmallIntType(BinOp->getType())) continue;

        bool ShouldPromote = false;
        SmallVector<Value*, 2> NewOperands;

        for (Value *Op : BinOp->operands()) {
          if (isI64Value(Op)) {
            NewOperands.push_back(Op);
            ShouldPromote = true;
          } else if (ExtendedValues.count(Op)) {
            NewOperands.push_back(ExtendedValues[Op]);
            ShouldPromote = true;
          } else if (Instruction *OpI = dyn_cast<Instruction>(Op)) {
            if (isSmallIntType(OpI->getType()) && isDefinitionPoint(OpI) &&
                !ExtendedValues.count(OpI)) {
              IRBuilder<> B(OpI->getNextNode());
              Value *ZExt = B.CreateZExt(OpI, I64Ty, OpI->getName() + ".zext");
              ExtendedValues[OpI] = ZExt;
              LocalChanged = true;
              NewOperands.push_back(ZExt);
              ShouldPromote = true;
            } else {
              NewOperands.push_back(Op);
            }
          } else if (Argument *ArgOp = dyn_cast<Argument>(Op)) {
            if (ExtendedValues.count(ArgOp)) {
              NewOperands.push_back(ExtendedValues[ArgOp]);
              ShouldPromote = true;
            } else {
              NewOperands.push_back(Op);
            }
          } else {
            NewOperands.push_back(Op);
          }
        }

        if (!ShouldPromote) continue;

        IRBuilder<> Builder(BinOp);
        for (Value *&Op : NewOperands) {
          if (!Op->getType()->isIntegerTy(64))
            Op = Builder.CreateZExt(Op, I64Ty, Op->getName() + ".zext");
        }

        Value *NewBinOp = nullptr;
        std::string OrigName = BinOp->getName().str();

        switch (BinOp->getOpcode()) {
          case Instruction::Add:  NewBinOp = Builder.CreateAdd(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::Sub:  NewBinOp = Builder.CreateSub(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::Mul:  NewBinOp = Builder.CreateMul(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::And:  NewBinOp = Builder.CreateAnd(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::Or:   NewBinOp = Builder.CreateOr(NewOperands[0], NewOperands[1], OrigName);  break;
          case Instruction::Xor:  NewBinOp = Builder.CreateXor(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::Shl:  NewBinOp = Builder.CreateShl(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::LShr: NewBinOp = Builder.CreateLShr(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::AShr: NewBinOp = Builder.CreateAShr(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::UDiv: NewBinOp = Builder.CreateUDiv(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::SDiv: NewBinOp = Builder.CreateSDiv(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::URem: NewBinOp = Builder.CreateURem(NewOperands[0], NewOperands[1], OrigName); break;
          case Instruction::SRem: NewBinOp = Builder.CreateSRem(NewOperands[0], NewOperands[1], OrigName); break;
          default: continue;
        }

        ExtendedValues[BinOp] = NewBinOp;
        PromotedBinOps.insert(BinOp);
        LocalChanged = true;
      }
    }

    // Phase 2b: Upgrade zext'd PHIs to full i64 PHIs.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        PHINode *Phi = dyn_cast<PHINode>(&I);
        if (!Phi) continue;
        if (!isSmallIntType(Phi->getType())) continue;
        if (PromotedPHIs.count(Phi)) continue;
        if (!ExtendedValues.count(Phi)) continue;
        if (!isa<ZExtInst>(ExtendedValues[Phi])) continue;

        auto getI64Form = [&](Value *Inc) -> Value* {
          if (isI64Value(Inc)) return Inc;
          if (ExtendedValues.count(Inc)) return ExtendedValues[Inc];
          if (TruncInst *Tr = dyn_cast<TruncInst>(Inc))
            if (isOurTrunc(Tr) && isI64Value(Tr->getOperand(0)))
              return Tr->getOperand(0);
          if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc))
            if (isSmallIntType(CI->getType()))
              return ConstantInt::get(I64Ty, CI->getZExtValue());
          return nullptr;
        };

        bool AllKnown = true;
        for (Value *Inc : Phi->incoming_values())
          if (!getI64Form(Inc)) { AllKnown = false; break; }
        if (!AllKnown) continue;

        IRBuilder<> Builder(Phi);
        PHINode *NewPhi = Builder.CreatePHI(I64Ty,
                                            Phi->getNumIncomingValues(),
                                            Phi->getName());
        for (unsigned i = 0, e = Phi->getNumIncomingValues(); i < e; ++i)
          NewPhi->addIncoming(getI64Form(Phi->getIncomingValue(i)),
                              Phi->getIncomingBlock(i));

        Value *OldZExt = ExtendedValues[Phi];
        OldZExt->replaceAllUsesWith(NewPhi);
        if (Instruction *OldZExtI = dyn_cast<Instruction>(OldZExt))
          if (OldZExtI->use_empty())
            OldZExtI->eraseFromParent();

        // Clear original PHI's incoming values so promoted BinOps feeding it
        // lose this use and become dead (use_empty) for erasure.
        Value *Poison = PoisonValue::get(Phi->getType());
        for (unsigned i = 0, e = Phi->getNumIncomingValues(); i < e; ++i)
          Phi->setIncomingValue(i, Poison);

        ExtendedValues[Phi] = NewPhi;
        PromotedPHIs.insert(Phi);
        LocalChanged = true;
      }
    }

    // Phase 2c: Fix up incoming values of already-promoted PHIs.
    for (PHINode *OldPhi : PromotedPHIs) {
      PHINode *NewPhi = cast<PHINode>(ExtendedValues[OldPhi]);
      for (unsigned i = 0, e = OldPhi->getNumIncomingValues(); i < e; ++i) {
        Value *OldInc = OldPhi->getIncomingValue(i);
        Value *CurInc = NewPhi->getIncomingValue(i);
        if (!isI64Value(CurInc)) continue;

        Value *Better = nullptr;
        if (ExtendedValues.count(OldInc)) {
          Value *Cand = ExtendedValues[OldInc];
          if (Cand->getType()->isIntegerTy(64) && Cand != CurInc)
            Better = Cand;
        }
        if (!Better) {
          if (TruncInst *Tr = dyn_cast<TruncInst>(OldInc))
            if (isOurTrunc(Tr) && isI64Value(Tr->getOperand(0)) &&
                Tr->getOperand(0) != CurInc)
              Better = Tr->getOperand(0);
        }
        if (Better) {
          NewPhi->setIncomingValue(i, Better);
          LocalChanged = true;
        }
      }
    }

    Changed |= LocalChanged;
  } while (LocalChanged);

  // Pre-Phase-3: fold pre-existing zexts whose source is now promoted.
  for (BasicBlock &BB : F) {
    SmallVector<Instruction*, 8> ToErase;
    for (Instruction &I : BB) {
      ZExtInst *ZExt = dyn_cast<ZExtInst>(&I);
      if (!ZExt || !ZExt->getDestTy()->isIntegerTy(64)) continue;
      if (isOurZExt(ZExt)) continue;
      Value *ZExtSrc = ZExt->getOperand(0);
      if (ExtendedValues.count(ZExtSrc)) {
        Value *Promoted = ExtendedValues[ZExtSrc];
        if (Promoted->getType()->isIntegerTy(64) && Promoted != ZExt) {
          ZExt->replaceAllUsesWith(Promoted);
          ToErase.push_back(ZExt);
          Changed = true;
        }
      }
    }
    for (Instruction *Dead : ToErase)
      if (Dead->use_empty())
        Dead->eraseFromParent();
  }

  // Phase 3: Replace uses and insert truncs where needed.
  DenseMap<Value*, Value*> TruncCache;

  for (auto &Pair : ExtendedValues) {
    Value *Original = Pair.first;
    Value *Extended = Pair.second;
    Type *OrigType = Original->getType();

    if (!isSmallIntType(OrigType)) continue;

    SmallVector<Use*, 16> UsesToReplace;
    SmallVector<Use*, 16> UsesNeedingTrunc;

    for (Use &U : Original->uses()) {
      User *UserInst = U.getUser();
      if (UserInst == Extended) continue;
      // Skip users that will be erased (promoted BinOps and promoted PHIs).
      if (BinaryOperator *UBO = dyn_cast<BinaryOperator>(UserInst))
        if (PromotedBinOps.count(UBO)) continue;
      if (PHINode *UPHI = dyn_cast<PHINode>(UserInst))
        if (PromotedPHIs.count(UPHI)) continue;

      if (userNeedsOriginalType(Original, U))
        UsesNeedingTrunc.push_back(&U);
      else
        UsesToReplace.push_back(&U);
    }

    for (Use *U : UsesToReplace)
      U->set(Extended);

    if (!UsesNeedingTrunc.empty()) {
      Instruction *ExtInst = dyn_cast<Instruction>(Extended);
      if (!ExtInst) continue;

      Value *Trunc = nullptr;
      auto It = TruncCache.find(Extended);
      if (It != TruncCache.end()) {
        Trunc = It->second;
      } else {
        IRBuilder<> Builder(ExtInst->getNextNode());
        Trunc = Builder.CreateTrunc(Extended, OrigType,
                                    Original->getName() + ".trunc");
        TruncCache[Extended] = Trunc;
      }
      for (Use *U : UsesNeedingTrunc)
        U->set(Trunc);
    }
  }

  // Erase dead promoted BinOps and PHIs.
  // RAUW(poison) first to break all cross-references between promoted
  // instructions, then erase unconditionally.
  {
    SmallVector<Instruction*, 16> Worklist;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I))
          if (PromotedBinOps.count(BO)) Worklist.push_back(BO);
        if (PHINode *PH = dyn_cast<PHINode>(&I))
          if (PromotedPHIs.count(PH)) Worklist.push_back(PH);
      }
    for (Instruction *I : Worklist) {
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
      Changed = true;
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
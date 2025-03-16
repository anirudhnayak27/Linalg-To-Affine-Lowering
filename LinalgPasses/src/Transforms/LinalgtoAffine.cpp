#include "Passes.h"  
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOAFFINEPASS
#include "Passes.h.inc"
} 

using namespace mlir;
using namespace mlir::linalg;

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals);
    affine::canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<affine::AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

template <typename LoadOpTy, typename StoreOpTy, typename OpType>
static void inlineRegionAndEmitStore(OpBuilder &b, Location loc, OpType op,
                                     ArrayRef<Value> indexedValues,
                                     ArrayRef<SmallVector<Value>> indexing,
                                     ArrayRef<Value> outputBuffers) {
  auto &block = op->getRegion(0).front();
  IRMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &opInst : block.without_terminator()) {
    auto *newOp = b.clone(opInst, map);
    map.map(opInst.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    b.create<StoreOpTy>(loc, toStore, outputBuffers[operand.getOperandNumber()],
                        indexing[operand.getOperandNumber()]);
  }
}

template <typename LoadOpTy, typename StoreOpTy>
static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs,
                                     LinalgOp linalgOp) {
  assert(linalgOp.hasPureBufferSemantics() &&
         "expected linalg op with buffer semantics");
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());
  auto allIvsPlusDims = SmallVector<Value>(allIvs);

  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<affine::AffineLoadOp>(loc, inputOperand->get(), indexing));
  }
  for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<affine::AffineLoadOp>(loc, outputOperand.get(), indexing));
  }

  SmallVector<SmallVector<Value>, 8> indexing;
  SmallVector<Value> outputBuffers;
  for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
    if (!isa<MemRefType>(outputOperand.get().getType()))
      continue;
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand), allIvsPlusDims));
    outputBuffers.push_back(outputOperand.get());
  }
  inlineRegionAndEmitStore<affine::AffineLoadOp, affine::AffineStoreOp>(
      b, loc, linalgOp, indexedValues, indexing, outputBuffers);
}

static void replaceIndexOpsByInductionVariables(RewriterBase &rewriter,
                                                LinalgOp linalgOp,
                                                ArrayRef<Operation *> loopOps) {
  SmallVector<Value> allIvs;
  for (Operation *loopOp : loopOps) {
    llvm::TypeSwitch<Operation *>(loopOp)
        .Case([&](affine::AffineForOp affineForOp) {
          allIvs.push_back(affineForOp.getInductionVar());
        })
        .Default([&](Operation *op) { assert(false && "unexpected op"); });
  }
  assert(linalgOp.getNumLoops() == allIvs.size() &&
         "expected the number of loops and induction variables to match");
  if (!loopOps.empty()) {
    auto loopOp = cast<LoopLikeOpInterface>(loopOps.back());
    for (Region *r : loopOp.getLoopRegions())
      for (IndexOp indexOp : llvm::make_early_inc_range(r->getOps<IndexOp>()))
        rewriter.replaceOp(indexOp, allIvs[indexOp.getDim()]);
  }
}

template <typename LoopTy = affine::AffineForOp>
static FailureOr<LinalgLoops> linalgOpToLoopsImpl(RewriterBase &rewriter,
                                                  LinalgOp linalgOp) {
  using LoadOpTy = affine::AffineLoadOp;
  using StoreOpTy = affine::AffineStoreOp;

  assert(linalgOp.hasPureBufferSemantics() &&
         "expected linalg op with buffer semantics");

  auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
  auto iteratorTypes = linalgOp.getIteratorTypesArray();

  SmallVector<Value> allIvs;
  GenerateLoopNest<affine::AffineForOp>::doit(
      rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
        assert(operandValuesToUse == linalgOp->getOperands() &&
               "expect operands are captured and not passed by loop argument");
        allIvs.append(ivs.begin(), ivs.end());
        emitScalarImplementation<affine::AffineLoadOp, affine::AffineStoreOp>(
            b, loc, allIvs, linalgOp);
        return scf::ValueVector{};
      });

  SetVector<Operation *> loopSet;
  for (Value iv : allIvs) {
    if (!iv)
      return failure();
    BlockArgument ivVal = dyn_cast<BlockArgument>(iv);
    if (!ivVal)
      return failure();
    loopSet.insert(ivVal.getOwner()->getParentOp());
  }
  LinalgLoops loops(loopSet.begin(), loopSet.end());
  replaceIndexOpsByInductionVariables(rewriter, linalgOp, loops);
  return loops;
}

namespace {
struct LinalgRewritePattern : public RewritePattern {
  LinalgRewritePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp || !linalgOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          op, "expected linalg op with buffer semantics");
    }
    if (failed(linalgOpToLoopsImpl<affine::AffineForOp>(rewriter, linalgOp)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct FoldAffineOp : public RewritePattern {
  FoldAffineOp(MLIRContext *context)
      : RewritePattern(affine::AffineApplyOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto affineApplyOp = cast<affine::AffineApplyOp>(op);
    auto map = affineApplyOp.getAffineMap();
    if (map.getNumResults() != 1 || map.getNumInputs() > 1)
      return failure();
    AffineExpr expr = map.getResult(0);
    if (map.getNumInputs() == 0) {
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, constExpr.getValue());
        return success();
      }
      return failure();
    }
    if (dyn_cast<AffineDimExpr>(expr) || dyn_cast<AffineSymbolExpr>(expr)) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    return failure();
  }
};

struct LowerToAffineLoopsPass
    : public PassWrapper<LowerToAffineLoopsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](Operation *op) {
      RewritePatternSet patterns(&getContext());
      patterns.add<LinalgRewritePattern>(&getContext());
      affine::AffineApplyOp::getCanonicalizationPatterns(patterns, &getContext());
      patterns.add<FoldAffineOp>(&getContext());
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    });
  }
};
} 

namespace mlir {
namespace linalg {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgToAffinePass() {
  return std::make_unique<LowerToAffineLoopsPass>();
}

} 
} 
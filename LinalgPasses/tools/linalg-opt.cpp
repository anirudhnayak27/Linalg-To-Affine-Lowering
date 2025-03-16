#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "Transforms/Passes.h"

using namespace mlir;

void samplePassPipeline(OpPassManager &pm) {
  pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  pm.addPass(mlir::linalg::createConvertLinalgToAffinePass());
  pm.addPass(createCanonicalizerPass());
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect,
                mlir::arith::ArithDialect,
                mlir::linalg::LinalgDialect,
                mlir::affine::AffineDialect,
                mlir::bufferization::BufferizationDialect,
                mlir::memref::MemRefDialect>();

  registerAllDialects(registry);
  registerAllPasses();
  
  MLIRContext context(registry);

  PassPipelineRegistration<> pipeline(
      "linalg-to-affine", "Lower Linalg ops to affine loops", samplePassPipeline);

  return asMainReturnCode(MlirOptMain(argc, argv, "Linalg to Affine optimizer", registry));
}
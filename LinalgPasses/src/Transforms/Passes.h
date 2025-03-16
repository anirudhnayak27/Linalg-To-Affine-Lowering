#ifndef CUSTOM_LINALG_TRANSFORMS_PASSES_H
#define CUSTOM_LINALG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinOps.h" 
namespace mlir {
namespace linalg {

std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgToAffinePass();

inline void registerConvertLinalgToAffinePass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertLinalgToAffinePass();
  });
}

inline void registerCustomLinalgPasses() {
  registerConvertLinalgToAffinePass();
}

} 
} 

#endif 
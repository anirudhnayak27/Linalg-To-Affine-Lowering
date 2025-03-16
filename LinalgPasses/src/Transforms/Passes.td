#ifndef LINALG_PASSES
#define LINALG_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToAffine : Pass<"linalg-to-affine">  {
  let summary = "Lower Linalg ops with buffer semantics to affine.for loop nests.";
  let description = [{
    This pass converts Linalg operations into a series of affine.for loops,
    making the control flow explicit and enabling further affine-specific optimizations.
  }];
  let argument = "convert-linalg-to-affine-loops";
}

#endif // LINALG_PASSES
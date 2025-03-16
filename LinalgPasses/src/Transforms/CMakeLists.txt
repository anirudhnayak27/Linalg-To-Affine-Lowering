set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)

add_public_tablegen_target(MLIRSampleTransformsIncGen)

add_mlir_library(SamplePasses
  LinalgtoAffine.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}
  DEPENDS
  MLIRSampleTransformsIncGen
  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
)
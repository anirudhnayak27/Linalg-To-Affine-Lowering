get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

find_package(MLIR REQUIRED CONFIG)

add_llvm_executable(sample-opt
  linalg-opt.cpp
)

llvm_update_compile_flags(sample-opt)

target_include_directories(sample-opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(sample-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRTransforms
  MLIRParser
  MLIRLinalgTransforms
  SamplePasses
)
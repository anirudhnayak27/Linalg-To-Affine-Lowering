#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace mlir;

void generateLinalgMatmul(MLIRContext &context) {
    // Create an empty MLIR module
    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
    OpBuilder builder(module->getBodyRegion());

    // Define tensor types
    Type f32 = builder.getF32Type();
    Type tensorA = RankedTensorType::get({4, 16}, f32);
    Type tensorB = RankedTensorType::get({16, 8}, f32);
    Type tensorC = RankedTensorType::get({4, 8}, f32);

    // Define function type: (tensorA, tensorB) -> tensorC
    FunctionType funcType = builder.getFunctionType({tensorA, tensorB}, {tensorC});
    auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(), "matmul", funcType);

    // Create function entry block
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder funcBuilder(entryBlock, entryBlock->begin());
    Value a = entryBlock->getArgument(0);
    Value b = entryBlock->getArgument(1);

    // Initialize output tensor
    Value cInit = funcBuilder.create<linalg::InitTensorOp>(builder.getUnknownLoc(),
                                                           ArrayRef<int64_t>{4, 8}, f32);

    // Perform matrix multiplication using linalg.matmul
    Value result = funcBuilder.create<linalg::MatmulOp>(builder.getUnknownLoc(), tensorC,
                                                         ValueRange{a, b}, ValueRange{cInit})
                                  .getResult(0);

    // Return the result tensor
    funcBuilder.create<func::ReturnOp>(builder.getUnknownLoc(), result);

    // Add function to the module
    module->push_back(funcOp);

    // Print the generated MLIR
    module->print(llvm::outs());
}

int main() {
    MLIRContext context;
    context.loadDialect<func::FuncDialect, linalg::LinalgDialect>();

    generateLinalgMatmul(context);

    return 0;
}

# Linalg-To-Affine-Lowering
Transformation Pass to lower linalg dialect to affine dialect

## Build Instruction 
```bash
$ mkdir build && cd build 
$ cmake ..
$ make -j32
```
### Testing 
```bash
./bin/sample-opt --linalg-to-affine ../test/input.mlir
```

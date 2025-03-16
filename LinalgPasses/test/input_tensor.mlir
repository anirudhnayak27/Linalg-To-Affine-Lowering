module {
  func.func @matmul(%A: tensor<4x16xf32>, %B: tensor<16x8xf32>) -> tensor<4x8xf32> {
    %C = linalg.matmul
      ins(%A, %B : tensor<4x16xf32>, tensor<16x8xf32>)
      -> tensor<4x8xf32>
    return %C : tensor<4x8xf32>
  }
}

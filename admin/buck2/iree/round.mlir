module @round {
  func.func @main(%weight: tensor<4xf32>, %noise: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = stablehlo.add %weight, %noise : tensor<4xf32>
    %observed = stablehlo.multiply %next, %next : tensor<4xf32>
    return %next, %observed : tensor<4xf32>, tensor<4xf32>
  }
}

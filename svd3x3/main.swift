//
//  main.swift
//  svd3x3
//
//  Created by Wayne Cochran on 1/15/25.
//

import Foundation
import simd

print("Hello, World!")

extension simd_float3x3 {
    func toString() -> String {
        var str : String = ""
        for r in 0 ..< 3 {
            str += NSString(format: "%+12.7f %+12.7f %+12.7f\n",
                            self[0][r], self[1][r], self[2][r]) as String
        }
        return str
    }
}

extension simd_float3 {
    func toString() -> String {
        return NSString(format: "%+12.7f %+12.7f %+12.7f",
                        self.x, self.y, self.z) as String
    }
}

func diagonalMatrix(from vector: simd_float3) -> simd_float3x3 {
    return simd_float3x3(
        simd_float3(vector.x, 0, 0), // First row
        simd_float3(0, vector.y, 0), // Second row
        simd_float3(0, 0, vector.z)  // Third row
    )
}


let matrix = simd_float3x3(
    simd_float3(1.0, 2.0, 3.0),
    simd_float3(4.0, 5.0, 6.0),
    simd_float3(7.0, 8.0, 9.0)
)

//
//From wolfram alpha
// https://www.wolframalpha.com/input?i=SVD+of+%7B%7B1%2C+4%2C+7%7D%2C+%7B2%2C+5%2C+8%7D%2C+%7B3%2C+6%2C+9%7D%7D
//U = (0.479671 | 0.776691 | 0.408248
//     0.572368 | 0.0756865 | -0.816497
//     0.665064 | -0.625318 | 0.408248)
//Σ = (16.8481 | 0       | 0
//     0       | 1.06837 | 0
//     0       | 0       | 0)
//V = (0.214837 | -0.887231 | 0.408248
//     0.520587 | -0.249644 | -0.816497
//     0.826338 | 0.387943 | 0.408248)

if let (U, S, V) = computeSVD(matrix: matrix) {
    print("U:\n\(U.toString())")
    print("S:\n\(S.toString())")
    print("V:\n\(V.toString())")

    let A = U * diagonalMatrix(from: S) * V.transpose
    print("matrix:\n\(matrix.toString())")
    print("A:\n\(A.toString())")
} else {
    print("Failed to compute SVD.")
}


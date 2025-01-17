//
//  SVD3x3.swift
//  svd3x3
//
//  Created by Wayne Cochran on 1/15/25.
//

import Foundation
import Accelerate
import simd

//
// Use the Accelerate framework to compute the Singular Value Decomposition of a 3x3 matrix.
//
func computeSVD(matrix: simd_float3x3) -> (U: simd_float3x3, S: simd_float3, V: simd_float3x3)? {
    var a = [
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2]
    ]

    var m = __CLPK_integer(3) // Number of rows
    var n = __CLPK_integer(3) // Number of columns
    var lda = m               // Leading dimension of A
    var ldu = m               // Leading dimension of U
    var ldvt = n              // Leading dimension of VT
    var info: __CLPK_integer = 0

    var s = [Float](repeating: 0.0, count: 3)              // Storage for the singular values
    var u = [Float](repeating: 0.0, count: Int(ldu * m))   // Storage for the left singular vectors (U)
    var vt = [Float](repeating: 0.0, count: Int(ldvt * n)) // Storage for the right singular vectors (VT)

    // 'A' => all columns of U / V^T are returned
    let job = ("A" as NSString).utf8String.map { UnsafeMutablePointer(mutating: $0) }

    // Workspace query (set lwork to -1 to get optimal size)
    var lwork = __CLPK_integer(-1)
    var workSizeQuery: Float = 0.0
    sgesvd_(job, job, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &workSizeQuery, &lwork, &info)

    // Allocate workspace
    lwork = __CLPK_integer(workSizeQuery)
    var work = [Float](repeating: 0.0, count: Int(lwork))

    // Compute SVD
    sgesvd_(job, job, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)

    if info != 0 {
        print("SVD computation failed with info = \(info)")
        return nil
    }

    // Convert results back to simd_float3x3
    var U = simd_float3x3(
        simd_float3(u[0], u[1], u[2]),
        simd_float3(u[3], u[4], u[5]),
        simd_float3(u[6], u[7], u[8])
    )

    let S = simd_float3(s[0], s[1], s[2])

    var V = simd_float3x3(
        simd_float3(vt[0], vt[1], vt[2]),
        simd_float3(vt[3], vt[4], vt[5]),
        simd_float3(vt[6], vt[7], vt[8])
    ).transpose // Transpose VT to get V

    // Ensure V is proper rotation matrix
    // Hope U will become one too :) -- it should be as long as
    // the input matrix does not contain any reflections.
    if V.determinant < 0 {
        V[2][0] = -V[2][0]
        V[2][1] = -V[2][1]
        V[2][2] = -V[2][2]

        U[2][0] = -U[2][0]
        U[2][1] = -U[2][1]
        U[2][2] = -U[2][2]

    }

    return (U: U, S: S, V: V)
}

//
// Use the Accelerate framework to compute the Singular Value Decomposition of a 3x3 matrix.
//
func computeSVD(matrix: simd_double3x3) -> (U: simd_double3x3, S: simd_double3, V: simd_double3x3)? {
    // Flatten the matrix into a column-major order array
    var a = [
        matrix[0][0], matrix[0][1], matrix[0][2], // First row (column-major layout)
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2]
    ]

    // SVD parameters
    var m = __CLPK_integer(3) // Number of rows
    var n = __CLPK_integer(3) // Number of columns
    var lda = m               // Leading dimension of A
    var ldu = m               // Leading dimension of U
    var ldvt = n              // Leading dimension of VT
    var info: __CLPK_integer = 0

    // Storage for singular values (S), U, and VT
    var s = [Double](repeating: 0.0, count: 3)              // Singular values
    var u = [Double](repeating: 0.0, count: Int(ldu * m))   // Left singular vectors (U)
    var vt = [Double](repeating: 0.0, count: Int(ldvt * n)) // Right singular vectors (VT)

    // 'A' => all columns of U / V^T are returned
    let job = ("A" as NSString).utf8String.map { UnsafeMutablePointer(mutating: $0) }

    // Workspace query (set lwork to -1 to get optimal size)
    var lwork = __CLPK_integer(-1)
    var workSizeQuery: Double = 0.0
    dgesvd_(job, job, // All rows of VT
            &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
            &workSizeQuery, &lwork, &info)

    // Allocate workspace based on the query
    lwork = __CLPK_integer(workSizeQuery)
    var work = [Double](repeating: 0.0, count: Int(lwork))

    // Perform SVD
    dgesvd_(job, job,
            &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
            &work, &lwork, &info)

    // Check for errors
    if info != 0 {
        print("SVD computation failed with info = \(info)")
        return nil
    }

    // Convert results back to simd_double3x3
    var U = simd_double3x3(
        simd_double3(u[0], u[1], u[2]), // First column of U
        simd_double3(u[3], u[4], u[5]), // Second column of U
        simd_double3(u[6], u[7], u[8])  // Third column of U
    )

    let S = simd_double3(s[0], s[1], s[2]) // Singular values

    var V = simd_double3x3(
        simd_double3(vt[0], vt[1], vt[2]), // First row of VT (transposed to column)
        simd_double3(vt[3], vt[4], vt[5]), // Second row of VT (transposed to column)
        simd_double3(vt[6], vt[7], vt[8])  // Third row of VT (transposed to column)
    ).transpose // Transpose VT to get V

    // Ensure V is a proper rotation matrix
    if V.determinant < 0 {
        V[2][0] = -V[2][0]
        V[2][1] = -V[2][1]
        V[2][2] = -V[2][2]

        U[2][0] = -U[2][0]
        U[2][1] = -U[2][1]
        U[2][2] = -U[2][2]
    }

    return (U: U, S: S, V: V)
}



import numpy as np


def gaussian_elimination(A, b):
    """
    Solve a linear system A*x = b using Gaussian elimination
    with partial pivoting.

    Parameters
    ----------
    A : array-like
        Coefficient matrix.

    b : array-like
        Right-hand-side vector.

    Returns
    -------
    dict
        Contains:
        - solution
        - upper_matrix
        - transformed_b
        - steps
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    rows, cols = A.shape

    if rows != cols:
        raise ValueError("A must be a square matrix.")

    if len(b) != rows:
        raise ValueError(
            "The length of b must match the number of rows in A."
        )

    U = A.copy()
    rhs = b.copy()

    steps = []

    n = rows

    # ---------------------------------------------------------
    # Forward elimination with partial pivoting
    # ---------------------------------------------------------

    for pivot_index in range(n - 1):

        # -----------------------------------------------------
        # 1. Find best pivot
        # -----------------------------------------------------

        pivot_row = (
            pivot_index
            + np.argmax(
                np.abs(
                    U[pivot_index:, pivot_index]
                )
            )
        )

        # -----------------------------------------------------
        # 2. Check pivot
        # -----------------------------------------------------

        if np.isclose(
            U[pivot_row, pivot_index],
            0.0,
        ):
            raise ValueError(
                "The system does not have a unique solution."
            )

        # -----------------------------------------------------
        # 3. Swap rows if necessary
        # -----------------------------------------------------

        if pivot_row != pivot_index:

            U[
                [pivot_index, pivot_row]
            ] = U[
                [pivot_row, pivot_index]
            ]

            rhs[
                [pivot_index, pivot_row]
            ] = rhs[
                [pivot_row, pivot_index]
            ]

            steps.append(
                {
                    "type": "swap",
                    "row_1": pivot_index,
                    "row_2": pivot_row,
                    "matrix": U.copy(),
                    "rhs": rhs.copy(),
                }
            )

        # -----------------------------------------------------
        # 4. Elimination
        # -----------------------------------------------------

        for row_index in range(
            pivot_index + 1,
            n,
        ):

            factor = (
                U[row_index, pivot_index]
                / U[pivot_index, pivot_index]
            )

            U[row_index, :] = (
                U[row_index, :]
                - factor * U[pivot_index, :]
            )

            rhs[row_index] = (
                rhs[row_index]
                - factor * rhs[pivot_index]
            )

            steps.append(
                {
                    "type": "elimination",
                    "pivot_row": pivot_index,
                    "target_row": row_index,
                    "factor": factor,
                    "matrix": U.copy(),
                    "rhs": rhs.copy(),
                }
            )

    # ---------------------------------------------------------
    # Back substitution
    # ---------------------------------------------------------

    solution = np.zeros(n)

    for row_index in range(n - 1, -1, -1):

        diagonal = U[
            row_index,
            row_index,
        ]

        if np.isclose(
            diagonal,
            0.0,
        ):
            raise ValueError(
                "The system does not have a unique solution."
            )

        known_terms = np.dot(
            U[
                row_index,
                row_index + 1:
            ],
            solution[
                row_index + 1:
            ],
        )

        solution[row_index] = (
            rhs[row_index]
            - known_terms
        ) / diagonal

    return {
        "solution": solution,
        "upper_matrix": U,
        "transformed_b": rhs,
        "steps": steps,
    }
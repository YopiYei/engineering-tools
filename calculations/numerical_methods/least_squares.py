import numpy as np
import pandas as pd


def linear_least_squares(x, y):
    """
    Linear regression using the explicit least-squares equations.

    The fitted straight line is:

        y = m*x + b

    where:
        m = slope
        b = y-intercept

    Parameters
    ----------
    x : array-like
        Independent variable values.

    y : array-like
        Dependent variable values.

    Returns
    -------
    dict
        Contains:
        - slope
        - intercept
        - correlation_coefficient
        - determination_coefficient
        - calculation_table
        - summations
    """

    # ---------------------------------------------------------
    # 1. Convert input data
    # ---------------------------------------------------------

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ---------------------------------------------------------
    # 2. Input validation
    # ---------------------------------------------------------

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of values.")

    if len(x) < 2:
        raise ValueError("At least two data points are required.")

    if np.all(x == x[0]):
        raise ValueError("All x values cannot be equal.")

    # ---------------------------------------------------------
    # 3. Number of observations
    # ---------------------------------------------------------

    n = len(x)

    # ---------------------------------------------------------
    # 4. Auxiliary columns
    # ---------------------------------------------------------

    xy = x * y
    x_squared = x**2
    y_squared = y**2

    # ---------------------------------------------------------
    # 5. Summations
    # ---------------------------------------------------------

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(xy)
    sum_x_squared = np.sum(x_squared)
    sum_y_squared = np.sum(y_squared)

    # ---------------------------------------------------------
    # 6. Least-squares denominator
    # ---------------------------------------------------------

    denominator = (n * sum_x_squared - sum_x**2)

    if denominator == 0:
        raise ValueError(
            "The least-squares denominator is zero."
        )

    # ---------------------------------------------------------
    # 7. Slope
    #
    #          n Σ(xy) - Σx Σy
    #    m = -------------------
    #          n Σ(x²) - (Σx)²
    # ---------------------------------------------------------

    slope = (n * sum_xy - sum_x * sum_y) / denominator

    # ---------------------------------------------------------
    # 8. Y-intercept
    #
    #        Σyi - m Σxi
    #    b = -----------
    #             n
    # ---------------------------------------------------------

    intercept = (sum_y - slope * sum_x) / n

    # ---------------------------------------------------------
    # 9. Predicted values
    #
    #    y_hat = m*x + b
    # ---------------------------------------------------------

    y_predicted = slope * x + intercept

    # ---------------------------------------------------------
    # 10. Residuals
    #
    #    beta = y - y_hat
    # ---------------------------------------------------------

    beta = y - y_predicted
    beta_squared = beta**2

    # ---------------------------------------------------------
    # 11. Correlation coefficient
    #
    #               nΣxy - ΣxΣy
    #    r = -----------------------------
    #        sqrt([nΣx²-(Σx)²][nΣy²-(Σy)²])
    # ---------------------------------------------------------

    correlation_denominator = np.sqrt(
        (
            n*sum_x_squared - sum_x**2
        )
        *
        (
            n * sum_y_squared
            - sum_y**2
        )
    )

    if correlation_denominator == 0:
        correlation_coefficient = np.nan
    else:
        correlation_coefficient = (
            n * sum_xy
            - sum_x * sum_y
        ) / correlation_denominator

    # ---------------------------------------------------------
    # 12. Coefficient of determination
    #
    #    R² = r²
    # ---------------------------------------------------------

    determination_coefficient = correlation_coefficient**2

    # ---------------------------------------------------------
    # 13. Calculation table
    # ---------------------------------------------------------

    calculation_table = pd.DataFrame(
        {
            "xi": x,
            "yi": y,
            "xi·yi": xy,
            "xi²": x_squared,
            "yi²": y_squared,
            "ŷi": y_predicted,
            "βi": beta,
            "βi²": beta_squared,
        }
    )

    # ---------------------------------------------------------
    # 14. Summations
    # ---------------------------------------------------------

    summations = {
        "n": n,
        "Σxi": sum_x,
        "Σyi": sum_y,
        "Σxi·yi": sum_xy,
        "Σxi²": sum_x_squared,
        "Σyi²": sum_y_squared,
        "Σβi²": np.sum(beta_squared),
    }

    # ---------------------------------------------------------
    # 15. Results
    # ---------------------------------------------------------

    return {
    "slope": slope,
    "intercept": intercept,
    "correlation_coefficient": correlation_coefficient,
    "determination_coefficient": determination_coefficient,
    "y_predicted": y_predicted,
    "beta": beta,
    "calculation_table": calculation_table,
    "summations": summations,
}
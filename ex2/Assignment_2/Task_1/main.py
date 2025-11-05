from sklearn import linear_model
import pandas as pd
import numpy as np


def main():
    # Read CSV file with data points
    df: pd.DataFrame = pd.read_csv("datapoints.csv")

    # Extract columns into separate lists
    X: list[list[float]] = df[["age", "area"]].values.tolist()
    y: list[float] = df["price"].tolist()

    ## Analytical solution
    print("########## Analytical solution ##########")
    # Calculate coefficients using the Normal Equation
    coeffs: np.ndarray = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y
    )
    print("Coefficients [age, area]:", coeffs)

    # Predict for a new data point
    age: int = 10
    area: float = 50.0
    predicted_y_analytical: float = coeffs[0] * age + coeffs[1] * area
    print(f"Prediction for {age=}, {area=} (analytical):", predicted_y_analytical)

    # Find the least squares error
    real_y: float = 427451.10
    print("Least Squares Error:", (predicted_y_analytical - real_y) ** 2)
    print("L1 Error:", np.abs(predicted_y_analytical - real_y))

    ## sklearn model
    print("\n\n########## Sklearn model ##########")

    # Calculate coefficients using the Normal Equation
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print("Coefficients [age, area]:", regr.coef_)

    # Predict for a new data point
    age: int = 10
    area: float = 50.0
    predicted_y: float = regr.predict([[age, area]])[0]
    print(f"Prediction for {age=}, {area=}:", predicted_y)

    # Find the least squares error
    real_y: float = 427451.10
    print("Least Squares Error:", (predicted_y - real_y) ** 2)
    print("L1 Error:", np.abs(predicted_y - real_y))


if __name__ == "__main__":
    main()

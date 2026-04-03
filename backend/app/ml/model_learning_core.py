import csv

from sklearn.linear_model import LinearRegression


def manual_learning_demo():
    print("MANUAL LEARNING DEMO (NO SKLEARN)")
    print("----------------------------------")
    print("Goal: learn price from size using only one weight.\n")

    # Simplified training data:
    # x = size in hundreds of sqft, y = price in lakh
    x_values = [10, 15, 20]
    y_values = [50, 75, 100]

    weight = 2.0
    bias = 0.0
    learning_rate = 0.8
    epochs = 8

    print(f"Start weight: {weight}")
    print(f"Start bias: {bias}\n")

    for epoch in range(1, epochs + 1):
        predictions = [(weight * x) + bias for x in x_values]
        errors = [actual - pred for actual, pred in zip(y_values, predictions)]
        loss = sum(abs(err) for err in errors) / len(errors)  # MAE

        avg_error = sum(errors) / len(errors)

        # Intuition:
        # avg_error > 0 means predictions are mostly too low -> increase weight
        # avg_error < 0 means predictions are mostly too high -> decrease weight
        weight = weight + (learning_rate * (avg_error / max(x_values)))

        print(
            f"Epoch {epoch}: weight={weight:.3f}, "
            f"loss(MAE)={loss:.3f}, avg_error={avg_error:.3f}"
        )

    final_predictions = [(weight * x) + bias for x in x_values]
    print("\nFinal predictions:", [round(v, 2) for v in final_predictions])
    print("Actual values    :", y_values)


def sklearn_demo():
    print("\nSKLEARN DEMO")
    print("------------")

    sizes = []
    prices = []

    with open("datasets/house_prices_sample.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sizes.append([int(row["size_sqft"])])  # feature X
            prices.append(float(row["price_lakh"]))  # label y

    model = LinearRegression()
    model.fit(sizes, prices)

    train_predictions = model.predict(sizes)
    train_mae = sum(abs(actual - pred) for actual, pred in zip(prices, train_predictions)) / len(
        prices
    )

    sample_size = 1600
    sample_prediction = model.predict([[sample_size]])[0]

    print("Learned weight (slope):", round(model.coef_[0], 4), "lakh per sqft")
    print("Learned bias (intercept):", round(model.intercept_, 3), "lakh")
    print("Training loss (MAE):", round(train_mae, 3), "lakh")
    print(f"Predicted price for {sample_size} sqft:", round(sample_prediction, 2), "lakh")


def main():
    manual_learning_demo()
    sklearn_demo()


if __name__ == "__main__":
    main()

def train_manual_model():
    # Simple training data where the real pattern is: y = 3 * x
    x_values = [1, 2, 3, 4, 5]
    y_actual = [3, 6, 9, 12, 15]

    weight = 0.5  # bad start on purpose
    learning_rate = 0.2
    epochs = 10
    loss_history = []

    print("Manual ML Learning (no libraries)")
    print("----------------------------------")
    print("We use one rule: prediction = weight * x")
    print(f"Start weight: {weight}\n")

    avg_x = sum(x_values) / len(x_values)

    for epoch in range(1, epochs + 1):
        predictions = [weight * x for x in x_values]
        errors = [actual - predicted for actual, predicted in zip(y_actual, predictions)]
        mae = sum(abs(err) for err in errors) / len(errors)
        avg_error = sum(errors) / len(errors)

        # If avg_error is positive, predictions are mostly low -> increase weight.
        # If avg_error is negative, predictions are mostly high -> decrease weight.
        weight_update = learning_rate * (avg_error / avg_x)
        weight += weight_update

        loss_history.append(mae)

        print(
            f"Epoch {epoch:>2}: weight={weight:.4f}, "
            f"avg_error={avg_error:.4f}, loss(MAE)={mae:.4f}"
        )

    final_predictions = [weight * x for x in x_values]
    print("\nFinal check")
    print("-----------")
    print("x values      :", x_values)
    print("actual y      :", y_actual)
    print("predicted y   :", [round(v, 3) for v in final_predictions])

    print("\nError trend (smaller is better)")
    print("-------------------------------")
    first_loss = loss_history[0]
    for i, loss in enumerate(loss_history, start=1):
        bar_len = max(1, int((loss / first_loss) * 25))
        print(f"Epoch {i:>2}: {'#' * bar_len} {loss:.4f}")


if __name__ == "__main__":
    train_manual_model()

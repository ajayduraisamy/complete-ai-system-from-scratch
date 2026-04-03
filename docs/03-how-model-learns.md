# 03 - How a Machine Learning Model Learns

## 1. Concept

Machine learning learning is a correction loop:

1. The model makes a prediction.
2. We compare prediction with the real answer.
3. The difference is the error.
4. The model updates itself to reduce that error.
5. It repeats this again and again.

For this step, we use one very simple rule:

- `prediction = w * x`

Where:

- `x` = input feature
- `w` = weight (how strongly x affects the prediction)

If `w` is wrong, predictions are wrong.  
Learning means adjusting `w` until predictions get close to actual values.

---

## 2. Visual explanation (clear)

Think of it like aiming at a target:

- Prediction = where your arrow lands
- Actual value = target center
- Error = distance from arrow to center

Text flow:

```text
Input x --> [Current weight w] --> Prediction
                  |
                  v
          Compare with Actual y
                  |
                  v
         Error = (Actual - Prediction)
                  |
                  v
         Update w a little and retry
```

When error keeps decreasing across iterations, the model is learning correctly.

---

## 3. Python code (no libraries)

File: `backend/app/ml/manual_learning.py`

```python
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
```

---

## 4. Output (step-by-step)

Command:

```bash
python backend/app/ml/manual_learning.py
```

Output:

```text
Manual ML Learning (no libraries)
----------------------------------
We use one rule: prediction = weight * x
Start weight: 0.5

Epoch  1: weight=1.0000, avg_error=7.5000, loss(MAE)=7.5000
Epoch  2: weight=1.4000, avg_error=6.0000, loss(MAE)=6.0000
Epoch  3: weight=1.7200, avg_error=4.8000, loss(MAE)=4.8000
Epoch  4: weight=1.9760, avg_error=3.8400, loss(MAE)=3.8400
Epoch  5: weight=2.1808, avg_error=3.0720, loss(MAE)=3.0720
Epoch  6: weight=2.3446, avg_error=2.4576, loss(MAE)=2.4576
Epoch  7: weight=2.4757, avg_error=1.9661, loss(MAE)=1.9661
Epoch  8: weight=2.5806, avg_error=1.5729, loss(MAE)=1.5729
Epoch  9: weight=2.6645, avg_error=1.2583, loss(MAE)=1.2583
Epoch 10: weight=2.7316, avg_error=1.0066, loss(MAE)=1.0066

Final check
-----------
x values      : [1, 2, 3, 4, 5]
actual y      : [3, 6, 9, 12, 15]
predicted y   : [2.732, 5.463, 8.195, 10.926, 13.658]

Error trend (smaller is better)
-------------------------------
Epoch  1: ######################### 7.5000
Epoch  2: #################### 6.0000
Epoch  3: ################ 4.8000
Epoch  4: ############ 3.8400
Epoch  5: ########## 3.0720
Epoch  6: ######## 2.4576
Epoch  7: ###### 1.9661
Epoch  8: ##### 1.5729
Epoch  9: #### 1.2583
Epoch 10: ### 1.0066
```

---

## 5. Final understanding

- Prediction is the model's current guess (`w * x`).
- Error is the gap between actual value and predicted value.
- Learning happens by repeatedly adjusting `w` using the error direction.
- Iteration by iteration, loss decreases, so predictions improve.
- This same core idea powers bigger ML models too.

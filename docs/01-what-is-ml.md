# 01 - What Is Machine Learning?

Machine Learning (ML) is a way to teach computers to find patterns from data and use those patterns to make decisions or predictions.

Instead of writing every rule by hand, we give examples (data), and the system learns useful behavior from those examples.

---

## Real-world example

Imagine a house-price app:

- Input data: house size, number of bedrooms, location score
- Output: estimated house price

If we give the system many past examples (houses + real selling prices), it can learn how these inputs affect price and then estimate prices for new houses.

---

## Dataset basics: Features and Labels

In ML, we usually split data into:

- **Features (X):** the input information used to make a prediction
- **Label (y):** the correct answer we want to predict

For a house-price dataset:

- Features: `size_sqft`, `bedrooms`
- Label: `price_lakh`

Small sample:

| size_sqft | bedrooms | price_lakh |
|-----------|----------|------------|
| 900       | 2        | 48         |
| 1100      | 2        | 56         |
| 1400      | 3        | 72         |
| 1700      | 3        | 86         |
| 2100      | 4        | 105        |

---

## Python example (working with ML-style data)

This is not model training yet. We only prepare ML-style data and inspect simple patterns.

```python
# ml_data_intro.py
data = [
    {"size_sqft": 900, "bedrooms": 2, "price_lakh": 48},
    {"size_sqft": 1100, "bedrooms": 2, "price_lakh": 56},
    {"size_sqft": 1400, "bedrooms": 3, "price_lakh": 72},
    {"size_sqft": 1700, "bedrooms": 3, "price_lakh": 86},
    {"size_sqft": 2100, "bedrooms": 4, "price_lakh": 105},
]

# Features (X) and label (y)
X = [[row["size_sqft"], row["bedrooms"]] for row in data]
y = [row["price_lakh"] for row in data]

print("Features (X):", X)
print("Labels (y):", y)

avg_price = sum(y) / len(y)
print("Average price (lakh):", round(avg_price, 2))

# Check one simple pattern: price per square foot
price_per_sqft = [row["price_lakh"] * 100000 / row["size_sqft"] for row in data]
print("Price per sqft (approx):", [round(v, 2) for v in price_per_sqft])
```

### Output

```text
Features (X): [[900, 2], [1100, 2], [1400, 3], [1700, 3], [2100, 4]]
Labels (y): [48, 56, 72, 86, 105]
Average price (lakh): 73.4
Price per sqft (approx): [5333.33, 5090.91, 5142.86, 5058.82, 5000.0]
```

---

## Why this step matters

Before training any ML model, we must clearly understand:

1. What is the input (features)?
2. What is the output (label)?
3. What patterns are present in the dataset?

This foundation makes future ML steps much easier and less confusing.

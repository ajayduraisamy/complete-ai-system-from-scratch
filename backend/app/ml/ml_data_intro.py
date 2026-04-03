data = [
    {"size_sqft": 900, "bedrooms": 2, "price_lakh": 48},
    {"size_sqft": 1100, "bedrooms": 2, "price_lakh": 56},
    {"size_sqft": 1400, "bedrooms": 3, "price_lakh": 72},
    {"size_sqft": 1700, "bedrooms": 3, "price_lakh": 86},
    {"size_sqft": 2100, "bedrooms": 4, "price_lakh": 105},
]

X = [[row["size_sqft"], row["bedrooms"]] for row in data]
y = [row["price_lakh"] for row in data]

print("Features (X):", X)
print("Labels (y):", y)

avg_price = sum(y) / len(y)
print("Average price (lakh):", round(avg_price, 2))

price_per_sqft = [row["price_lakh"] * 100000 / row["size_sqft"] for row in data]
print("Price per sqft (approx):", [round(v, 2) for v in price_per_sqft])

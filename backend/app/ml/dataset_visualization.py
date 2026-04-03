import csv
import os

import matplotlib.pyplot as plt


DATASET_PATH = "datasets/house_prices_sample.csv"
PLOT_PATH = "docs/images/02-dataset-visualization.png"


def load_dataset(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "house_id": row["house_id"],
                    "size_sqft": int(row["size_sqft"]),
                    "bedrooms": int(row["bedrooms"]),
                    "location_score": float(row["location_score"]),
                    "price_lakh": float(row["price_lakh"]),
                }
            )
    return rows


def print_table(rows):
    header = f"{'house_id':<8} {'size_sqft':<10} {'bedrooms':<9} {'location_score':<15} {'price_lakh':<10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['house_id']:<8} {row['size_sqft']:<10} {row['bedrooms']:<9} "
            f"{row['location_score']:<15} {row['price_lakh']:<10}"
        )


def print_dataset_summary(rows):
    features = ["size_sqft", "bedrooms", "location_score"]
    label = "price_lakh"
    prices = [row["price_lakh"] for row in rows]

    print("\nDataset understanding")
    print("---------------------")
    print("Total rows:", len(rows))
    print("Features (X):", features)
    print("Label (y):", label)
    print("Price range (lakh):", min(prices), "to", max(prices))
    print("Average price (lakh):", round(sum(prices) / len(prices), 2))


def plot_patterns(rows, output_path: str):
    sizes = [row["size_sqft"] for row in rows]
    bedrooms = [row["bedrooms"] for row in rows]
    location_scores = [row["location_score"] for row in rows]
    prices = [row["price_lakh"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(sizes, prices, color="#1f77b4", s=70)
    axes[0].set_title("Size vs Price")
    axes[0].set_xlabel("Size (sqft)")
    axes[0].set_ylabel("Price (lakh)")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(location_scores, prices, color="#ff7f0e", s=70)
    axes[1].set_title("Location Score vs Price")
    axes[1].set_xlabel("Location score (1-10)")
    axes[1].set_ylabel("Price (lakh)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("House Dataset: Visual Pattern Check", fontsize=13)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    rows = load_dataset(DATASET_PATH)
    print("Table view of dataset")
    print("---------------------")
    print_table(rows)
    print_dataset_summary(rows)
    plot_patterns(rows, PLOT_PATH)
    print("\nVisualization saved to:", PLOT_PATH)


if __name__ == "__main__":
    main()

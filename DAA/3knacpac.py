class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
    def value_to_weight_ratio(self):
        return self.value / self.weight


def fractional_knapsack(items, capacity):
    items.sort(key=lambda x: x.value_to_weight_ratio(), reverse=True)

    total_value = 0
    for item in items:
        if capacity >= item.weight:
            capacity -= item.weight
            total_value += item.value
        else:
            fraction = capacity / item.weight
            total_value += item.value * fraction
            capacity = 0  # Knapsack is now full
            break

    return total_value

# Example usage
items = [
    Item(60, 10),  # Value = 60, Weight = 10
    Item(100, 20), # Value = 100, Weight = 20
    Item(120, 30)  # Value = 120, Weight = 30
]

capacity = 50
max_value = fractional_knapsack(items, capacity)
print(f"Maximum value in the knapsack: {max_value}")
# Convert the results .CVS file to a plot

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("ratio_results.csv")

# Plot the line graph
plt.figure(figsize=(8,6))
plt.plot(df["mix_ratio"], df["accuracy"], marker="o", linestyle="-", color="b", label="Accuracy")

# Add labels and title
plt.xlabel("Mix Ratio")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Mix Ratio")
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

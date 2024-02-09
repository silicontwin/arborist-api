################################################################################
# Attempt to recreate the box plots with partial violins from the paper:
# "A dual-system, machine-learning approach reveals how daily pubertal hormones
# relate to psychological well-being in everyday life"

# Nick Warren
# Feb. 8, 2024
################################################################################

library(ggplot2)

# Import the iris dataset
data(iris)

# Create the box/violin plots
p <- ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_violin(trim = TRUE) +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  geom_jitter(width = 0.1, size = 1, alpha = 0.5)

# Add a white rectangle to cover the right half of the violin plot (adjust xmin/xmax to cover the correct area)
p <- p + annotate("rect", xmin = 1 - 0.05, xmax = 1 + 0.5, ymin = min(iris$Sepal.Length), ymax = max(iris$Sepal.Length), fill = "white") +
  annotate("rect", xmin = 2 - 0.05, xmax = 2 + 0.5, ymin = min(iris$Sepal.Length), ymax = max(iris$Sepal.Length), fill = "white") +
  annotate("rect", xmin = 3 - 0.05, xmax = 3 + 0.5, ymin = min(iris$Sepal.Length), ymax = max(iris$Sepal.Length), fill = "white")

# Adjust the position of the boxplot to align with the violin plots
p <- p + geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, position = position_nudge(x = -0.05))

# Print the plot
print(p)

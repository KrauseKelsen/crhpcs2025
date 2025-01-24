library(dplyr)
library(mvtnorm)

generateGaussianData <- function(n, center, sigma, label) {
  data = rmvnorm(n, mean = center, sigma = sigma)
  data = data.frame(data)
  names(data) = c("x", "y")
  data = data %>% mutate(class=factor(label))
  data
}

dataset1 <- {
  # cluster 1
  n = 5000
  center = c(5, 5)
  sigma = matrix(c(1, 0, 0, 1), nrow = 2)
  data1 = generateGaussianData(n, center, sigma, 1)
  # cluster 2
  n = 5000
  center = c(1, 1)
  sigma = matrix(c(1, 0, 0, 1), nrow = 2)
  data2 = generateGaussianData(n, center, sigma, 2)
  # all data
  data = bind_rows(data1, data2)
  data$dataset = "1 - Mixture of Gaussians"
  data
}

library(ggplot2)
dataset1 %>% ggplot(aes(x=x, y=y, color=class)) +
  geom_point() +
  coord_fixed() +
  scale_shape_manual(values=c(2, 3))

ggsave("plot.png")

initial_dataset <- dataset1[, c("x", "y")]

write.csv(initial_dataset, file="dataset.csv", row.names = FALSE)

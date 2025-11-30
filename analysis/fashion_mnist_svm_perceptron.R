# Fashion MNIST Classification: Perceptron & SVM
# Author: Divyanshi Mishra

# Required packages
# install.packages("simpleNeural")
# install.packages("e1071")

library(simpleNeural)
library(e1071)

# 1. Load data
load("fashion_mnist.RData")

# 2. Inspect size
cat("Training samples:", nrow(train), "\n")
cat("Testing samples:", nrow(test), "\n")

# 3. Label distribution
table(train$label)

# 4. Convert labels to factors
train$label <- as.factor(train$label)
test$label  <- as.factor(test$label)

# Separate features and labels
train_x <- train[, -1]
train_y <- train$label

test_x  <- test[, -1]
test_y  <- test$label

# -------------------------------------------------------
# 5. PERCEPTRON MODEL
# -------------------------------------------------------

set.seed(123)
perc_model <- sN.MLPtrain(
  X = train_x,
  Y = train_y,
  size = 20,        # hidden units
  maxit = 20        # number of iterations
)

perc_pred <- sN.MLPpredict(perc_model, test_x)

# Misclassification rate
perc_mis <- mean(perc_pred != test_y)
cat("Perceptron Misclassification Rate:", round(perc_mis, 4), "\n")

# Confusion matrix
perc_conf <- table(Actual = test_y, Predicted = perc_pred)
print(perc_conf)

# -------------------------------------------------------
# 6. SVM MODEL
# -------------------------------------------------------

svm_model <- svm(
  x = train_x,
  y = train_y,
  kernel = "linear",
  cost = 1,
  scale = FALSE
)

svm_pred <- predict(svm_model, test_x)

svm_mis <- mean(svm_pred != test_y)
cat("SVM Misclassification Rate:", round(svm_mis, 4), "\n")

svm_conf <- table(Actual = test_y, Predicted = svm_pred)
print(svm_conf)

# -------------------------------------------------------
# 7. IMPROVING ACCURACY (Optional Tuning)
# -------------------------------------------------------

# Try different perceptron settings
hidden_units <- c(10, 20, 30)
iterations    <- c(10, 20, 40)

results <- data.frame()

for (h in hidden_units) {
  for (it in iterations) {
    model <- sN.MLPtrain(train_x, train_y, size = h, maxit = it)
    pred  <- sN.MLPpredict(model, test_x)
    mis   <- mean(pred != test_y)

    results <- rbind(results,
                     data.frame(Hidden = h, Iterations = it, MisRate = mis))
  }
}

print(results)

# Try SVM with multiple kernels
kernels <- c("linear", "polynomial", "radial")
svm_results <- data.frame()

for (k in kernels) {
  model <- svm(train_x, train_y, kernel = k, cost = 1, scale = FALSE)
  pred  <- predict(model, test_x)
  mis   <- mean(pred != test_y)

  svm_results <- rbind(svm_results,
                       data.frame(Kernel = k, MisRate = mis))
}

print(svm_results)

# -------------------------------------------------------
# 8. Which classes are easy or difficult?
# -------------------------------------------------------

# Per-digit error for SVM (example)
digit_error <- 1 - diag(svm_conf) / rowSums(svm_conf)
print(digit_error)

# -------------------------------------------------------
# 9. OPTIONAL: Visualize a random training image
# -------------------------------------------------------

show_random_image <- function(index) {
  img <- train[index, -1]
  matrix_img <- matrix(as.numeric(img), nrow = 28, byrow = TRUE)
  image(t(apply(matrix_img, 2, rev)), col = gray.colors(255),
        main = paste("Label:", train$label[index]))
}

# Example:
# show_random_image(42)

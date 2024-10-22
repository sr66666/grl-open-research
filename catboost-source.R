
library(catboost)
library(caret)
library(readr)
library(pROC)  
library(showtext)


font_path <- "C:/Windows/Fonts/times.ttf"
font_add("Times New Roman", regular = font_path)
showtext_auto()


set.seed(123)


data <- read.csv("D:/data/volcanic-catboost.csv")


data <- data[, -(1:2)]


data$type <- as.factor(data$type)


trainIndex <- createDataPartition(data$type, p = 0.9, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


train_pool <- catboost.load_pool(data = train[, -which(names(train) == "type")], label = as.numeric(train$type) - 1)
test_pool <- catboost.load_pool(data = test[, -which(names(test) == "type")], label = as.numeric(test$type) - 1)

数
catboost_grid <- expand.grid(depth = c(4, 6),  
                             learning_rate = c(0.01, 0.1),  
                             iterations = seq(50, 200, by = 50), 
                             l2_leaf_reg = c(1, 3),  
                             rsm = c(0.8, 0.9),  
                             border_count = c(32, 64)) 

catboost_fit <- train(x = train[, -which(names(train) == "type")], y = train$type, method = catboost.caret,
                      trControl = trainControl(method = "cv", number = 10),  
                      tuneGrid = catboost_grid
)


best_params <- catboost_fit$bestTune
best_depth <- best_params$depth
best_learning_rate <- best_params$learning_rate
best_iterations <- best_params$iterations
best_l2_leaf_reg <- best_params$l2_leaf_reg
best_rsm <- best_params$rsm
best_border_count <- best_params$border_count


catboost_params <- list(depth = best_depth, learning_rate = best_learning_rate, iterations = best_iterations, l2_leaf_reg = best_l2_leaf_reg, rsm = best_rsm, border_count = best_border_count, loss_function = "MultiClass")
catboost_model <- catboost.train(train_pool, params = catboost_params)


predictions <- catboost.predict(catboost_model, test_pool, prediction_type = "Class")

predictions <- levels(train$type)[as.numeric(predictions) + 1]


accuracy <- mean(predictions == test$type)
cat("Accuracy:", accuracy, "\n")


confusionMatrix(factor(predictions, levels = levels(test$type)), test$type)
statistics_by_class <- confusionMatrix(factor(predictions, levels = levels(test$type)), test$type)$byClass


print(statistics_by_class)


write.csv(as.matrix(statistics_by_class), file = "D:/data/catboost_type_xingenng.csv")


predictions_prob <- catboost.predict(catboost_model, test_pool, prediction_type = "Probability")


roc_curves <- list()
auc_values <- numeric(length(levels(test$type)))  
for (i in 1:length(levels(test$type))) {
  true_labels <- as.numeric(test$type == levels(test$type)[i])
  predicted_probs <- predictions_prob[, i]
  
  roc_curves[[i]] <- roc(true_labels, predicted_probs)
  auc_values[i] <- auc(roc_curves[[i]])
  
  cat("Class:", levels(test$type)[i], "AUC:", auc_values[i], "\n")
}


mean_auc <- mean(auc_values)
cat("Mean AUC:", mean_auc, "\n")


par(family = "Times New Roman", mar = c(5, 5, 4, 2) + 0.1)


plot(roc_curves[[1]], main = NULL, cex.main = 2, cex.lab = 2, cex.axis = 1.2)


for (i in 2:length(roc_curves)) {
  lines(roc_curves[[i]], col = rainbow(length(roc_curves))[i], lwd = 2)
}


abline(h = mean_auc, col = "blue", lwd = 2)


text(x = max(roc_curves[[1]]$sensitivities), y = 0.76, labels = paste("Mean AUC:", round(mean_auc, 2)), pos = 2, col = "blue", cex = 1.5)


new_data <- read.csv("D:/data/GBRtype.csv")


new_data <- new_data[, -c(1, 2)]


new_data_pool <- catboost.load_pool(data = new_data)


new_predictions <- catboost.predict(catboost_model, new_data_pool, prediction_type = "Class")
new_predictions <- levels(train$type)[as.numeric(new_predictions) + 1]


new_predictions_prob <- catboost.predict(catboost_model, new_data_pool, prediction_type = "Probability")


print(new_predictions)
write.csv(new_predictions, file = "D:/data/gbrtyperfpre_catboost.csv")


print(new_predictions_prob)
write.csv(new_predictions_prob, file = "D:/data/gbrtyperfpre_prob_catboost.csv")


library(catboost)
library(caret)
library(readr)


set.seed(123)


data <- read.csv("D:/data/LIRFDATA-catboost.csv")


data <- data[, -1]


data$Li <- as.factor(data$Li)


trainIndex <- createDataPartition(data$Li, p = 0.9, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


train_pool <- catboost.load_pool(data = train[, -which(names(train) == "Li")], label = as.numeric(train$Li) - 1)
test_pool <- catboost.load_pool(data = test[, -which(names(test) == "Li")], label = as.numeric(test$Li) - 1)


catboost_grid <- expand.grid(depth = c(4, 6, 8),
                             learning_rate = c(0.01, 0.1, 0.2),
                             iterations = seq(100, 1000, by = 100),
                             l2_leaf_reg = c(1, 3, 5),
                             rsm = c(0.8, 0.9, 1.0),  
                             border_count = c(32, 64, 128))  

catboost_fit <- train(x = train[, -which(names(train) == "Li")], y = train$Li, method = catboost.caret,
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

predictions <- levels(train$Li)[as.numeric(predictions) + 1]


accuracy <- mean(predictions == test$Li)
cat("Accuracy:", accuracy, "\n")


confusionMatrix(factor(predictions, levels = levels(test$Li)), test$Li)
statistics_by_class <- confusionMatrix(factor(predictions, levels = levels(test$Li)), test$Li)$byClass


print(statistics_by_class)


write.csv(as.matrix(statistics_by_class), file = "D:/data/catboost_li_xingenng.csv")


new_data <- read.csv("D:/data/GBRCLEAN.csv")


new_data <- new_data[, -c(1, 2)]


new_data_pool <- catboost.load_pool(data = new_data)


new_predictions <- catboost.predict(catboost_model, new_data_pool, prediction_type = "Class")
new_predictions <- levels(train$Li)[as.numeric(new_predictions)+1]

print(new_predictions)
write.csv(new_predictions, file = "D:/data/gbrlirfpre_catboost.csv")

# 导入所需的库
library(e1071)
library(caret)
library(readr)
library(extrafont)
library(pROC)  
library(showtext)

set.seed(123)


data <- read.csv("D:/data/volcanic.csv")


data <- data[, -(1:2)]


data$type <- as.factor(data$type)


trainIndex <- createDataPartition(data$type, p = 0.9, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


svm_grid <- expand.grid(C = c(0.01, 0.1, 1, 10),  
                        sigma = c(0.01, 0.1, 0.5))  

svm_fit <- train(type ~ ., data = train, method = "svmRadial",
                 trControl = trainControl(method = "cv", number = 5), 
                 tuneGrid = svm_grid)

plot(svm_fit)

best_params <- svm_fit$bestTune
best_C <- best_params$C
best_sigma <- best_params$sigma


svm_model <- svm(type ~ ., data = train, kernel = "radial", cost = best_C, gamma = best_sigma, probability = TRUE)


predictions <- predict(svm_model, test, probability = TRUE)


accuracy <- mean(predictions == test$type)
cat("Accuracy:", accuracy, "\n")



confusionMatrix(predictions, test$type)
statistics_by_class <- confusionMatrix(predictions, test$type)$byClass


print(statistics_by_class)


write.csv(as.matrix(statistics_by_class), file = "D:/data/svm_type_xingneng.csv")


new_data <- read.csv("D:/data/GBRtype.csv")


new_data <- new_data[, -c(1, 2)]


new_predictions <- predict(svm_model, new_data, probability = TRUE)
new_predictions_prob <- attr(new_predictions, "probabilities")


print(new_predictions)
print(new_predictions_prob)


write.csv(new_predictions, file = "D:/data/gbrtypepre_svm.csv")
write.csv(new_predictions_prob, file = "D:/data/gbrtypepre_svm_prob.csv")

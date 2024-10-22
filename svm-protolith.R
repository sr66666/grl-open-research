
library(e1071)
library(caret)
library(readr)


set.seed(123)


data <- read.csv("D:/data/RFSOURCE.csv")


data <- data[, -1]


data$ROCK.NAME <- as.factor(data$ROCK.NAME)


trainIndex <- createDataPartition(data$ROCK.NAME, p = 0.9, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


svm_grid <- expand.grid(C = c(0.1, 1, 10),
                        sigma = c(0.1, 0.5, 1))

svm_fit <- train(ROCK.NAME ~ ., data = train, method = "svmRadial",
                 trControl = trainControl(method = "cv", number = 10),
                 tuneGrid = svm_grid)


best_params <- svm_fit$bestTune
best_C <- best_params$C
best_sigma <- best_params$sigma


svm_model <- svm(ROCK.NAME ~ ., data = train, kernel = "radial", cost = best_C, gamma = best_sigma)


predictions <- predict(svm_model, test)


accuracy <- mean(predictions == test$ROCK.NAME)
cat("Accuracy:", accuracy, "\n")


confusionMatrix(predictions, test$ROCK.NAME)
statistics_by_class <- confusionMatrix(predictions, test$ROCK.NAME)$byClass


print(statistics_by_class)


write.csv(as.matrix(statistics_by_class), file = "D:/data/svm_yanxing_xingneng.csv")


new_data <- read.csv("D:/data/GBRCLEAN.csv")


new_data <- new_data[, -c(1, 2)]


new_predictions <- predict(svm_model, new_data)

print(new_predictions)
write.csv(new_predictions, file = "D:/data/gbrpre_svm.csv")
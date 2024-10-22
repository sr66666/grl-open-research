
library(randomForest)
library(readr)
library(caret)
library(ggplot2)
library(pROC)  
library(showtext)


font_path <- "C:/Windows/Fonts/times.ttf"
font_add("Times New Roman", regular = font_path)
showtext_auto()


set.seed(123)


data <- read.csv("D:/data/LIRFDATA.csv")


data <- data[, -1]


data$Li <- as.factor(data$Li)


trainIndex <- createDataPartition(data$Li, p = 0.9, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


rf_grid <- expand.grid(mtry = seq(1, 13, by = 1)
                       )

rf_fit <- train(Li ~ ., data = train, method = "rf",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = rf_grid)
plot(rf_fit)

best_params <- rf_fit$bestTune
best_mtry <- best_params$mtry



rf_model <- randomForest(Li ~ ., data = train, mtry = best_mtry, ntree = 500)


predictions <- predict(rf_model, test)


accuracy <- mean(predictions == test$Li)
cat("Accuracy:", accuracy, "\n")


importance <- importance(rf_model)
varImpPlot(rf_model)


confusionMatrix(predictions, test$Li)
statistics_by_class <- confusionMatrix(predictions, test$Li, )$byClass


print(statistics_by_class)


write.csv(as.matrix(statistics_by_class), file = "D:/data/rf_li_xingneng.csv")


new_data <- read.csv("D:/data/GBRCLEAN.csv")


new_data <- new_data[, -c(1, 2)]


new_predictions <- predict(rf_model, new_data)

print(new_predictions)
write.csv(new_predictions, file = "D:/data/lipre-RF.csv")

prediction_probabilities <- predict(rf_model, new_data, type = "prob")
prediction_probabilities

write.csv(prediction_probabilities, file = "D:/data/Li_probability-RF.csv")



predictions_prob <- predict(rf_model, test, type = "prob")

roc_curves <- list()
auc_values <- numeric(length(levels(test$Li)))  
for (i in 1:length(levels(test$Li))) {
  true_labels <- as.numeric(test$Li == levels(test$Li)[i])
  predicted_probs <- predictions_prob[, i]
  
  roc_curves[[i]] <- roc(true_labels, predicted_probs)
  auc_values[i] <- auc(roc_curves[[i]])
  
  cat("Class:", levels(test$Li)[i], "AUC:", auc_values[i], "\n")
}


mean_auc <- mean(auc_values)
cat("Mean AUC:", mean_auc, "\n")


par(family = "Times New Roman", mar = c(5, 5, 4, 2) + 0.1)


plot(roc_curves[[1]], main = NULL, cex.main = 2, cex.lab = 2, cex.axis = 1.2)


for (i in 2:length(roc_curves)) {
  lines(roc_curves[[i]], col = rainbow(length(roc_curves))[i], lwd = 2)
}


abline(h = mean_auc, col = "blue", lwd = 2)


text(x = max(roc_curves[[1]]$sensitivities), y = 1 - min(roc_curves[[1]]$specificities), labels = paste("Mean AUC:", round(mean_auc, 2)), pos = 2, col = "blue", cex = 1.5)











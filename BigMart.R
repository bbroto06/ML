###############################################################################################################################################################
## Problem Statement
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. 
# Also, certain attributes of each product and store have been defined. 
# The aim is to build a predictive model and find out the sales of each product at a particular store.
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

## Please note that the data may have missing values as some stores might not report all the data due to technical glitches. 
# Hence, it will be required to treat them accordingly.

## Data
# We have train (8523) and test (5681) data set, train data set has both input and output variable(s). You need to predict the sales for test data set.

# Variable                  |     Description
# Item_Identifier           |   Unique product ID
# Item_Weight               |   Weight of product
# Item_Fat_Content          |   Whether the product is low fat or not
# Item_Visibility           |   The % of total display area of all products in a store allocated to the particular product
# Item_Type                 |   The category to which the product belongs
# Item_MRP                  |   Maximum Retail Price (list price) of the product
# Outlet_Identifier         |   Unique store ID
# Outlet_Establishment_Year |   The year in which store was established
# Outlet_Size               |   The size of the store in terms of ground area covered
# Outlet_Location_Type      |   The type of city in which the store is located
# Outlet_Type               |   Whether the outlet is just a grocery store or some sort of supermarket
# Item_Outlet_Sales         |   Sales of the product in the particulat store. This is the outcome variable to be predicted.

## Evaluation Metric:
# Your model performance will be evaluated on the basis of your prediction of the sales for the test data (test.csv), 
# which contains similar data-points as train except for the sales to be predicted. 
# Your submission needs to be in the format as shown in "SampleSubmission.csv".

# We at our end, have the actual sales for the test dataset, against which your predictions will be evaluated. 
# We will use the Root Mean Square Error value to judge your response.
# RMSE = Square root ( Sum of i=1 to N (Predicted - Actual)square / N )

# Where,
# N: total number of observations
# Predicted: the response entered by user
# Actual: actual values of sales

# Also, note that the test data is further divided into Public (25%) and Private (75%) data. 
# Your initial responses will be checked and scored on the Public data. 
# But, the final rankings will be based on score on Private data set. 
# Since this is a practice problem, we will keep declare winners after specific time intervals and refresh the competition.
###############################################################################################################################################################

# Clear environment
rm(list = ls(all=TRUE))

# Load required R library
library(MASS)
library(car)
library(corrplot)
library(stringr)
library(vegan)
library(ROCR)
library(zoo)
library(dplyr)
library(plyr)
library(TTR)
library(forecast)
library(DMwR)
library(data.table)
library(caret)
library(lattice)
library(ggplot2)
library(grid)
library(randomForest)
library(dummies)
library(FSelector)

# Set working directory
setwd("C://Users//brbhatta//Desktop//INSOFE//Practise//AnalyticsVidhya//PracticeProblem//Big_Mart_Sales_III")

# Read the dataset
original_data <- read.csv(file = "Train.csv", header = TRUE)
test_data <- read.csv(file = "Test.csv", header = TRUE)

# Explore and understand the data
head(original_data)
str(original_data)
summary(original_data)


head(test_data)
str(test_data)
summary(test_data)

###################################### Train.csv ################################################
# Replace "LF", "low fat" with -> "Low Fat" in Item_Fat_Content column
original_data <- original_data %>% mutate(Item_Fat_Content = replace(Item_Fat_Content, which(Item_Fat_Content == "LF" | Item_Fat_Content == "low fat"), "Low Fat"))

# Replace "reg" with -> "Regular" in Item_Fat_Content column
original_data <- original_data %>% mutate(Item_Fat_Content = replace(Item_Fat_Content, which(Item_Fat_Content == "reg"), "Regular"))

# Factor Item_Fat_Content column
original_data$Item_Fat_Content <- factor(original_data$Item_Fat_Content)

# Replace blank with NA in Outlet_Size column
original_data <- original_data %>% mutate(Outlet_Size = replace(Outlet_Size, which(Outlet_Size == ""), "NA"))

# Factor Outlet_Size column
original_data$Outlet_Size <- factor(original_data$Outlet_Size)



#################################### Test.csv ######################################################
# Replace "LF", "low fat" with -> "Low Fat" in Item_Fat_Content column
test_data <- test_data %>% mutate(Item_Fat_Content = replace(Item_Fat_Content, which(Item_Fat_Content == "LF" | Item_Fat_Content == "low fat"), "Low Fat"))

# Replace "reg" with -> "Regular" in Item_Fat_Content column
test_data <- test_data %>% mutate(Item_Fat_Content = replace(Item_Fat_Content, which(Item_Fat_Content == "reg"), "Regular"))

# Factor Item_Fat_Content column
test_data$Item_Fat_Content <- factor(test_data$Item_Fat_Content)

# Replace blank with NA in Outlet_Size column
test_data <- test_data %>% mutate(Outlet_Size = replace(Outlet_Size, which(Outlet_Size == ""), "NA"))

# Factor Outlet_Size column
test_data$Outlet_Size <- factor(test_data$Outlet_Size)



# Divide the original training data into train (70%) and validation (30%)
set.seed(987)
index_id<- createDataPartition(original_data$Item_Outlet_Sales, p = 0.7, list = F)
pre_train <- original_data[index_id,]
pre_validation <- original_data[-index_id,]

# check summary of pre_train and pre_validation
summary(pre_train)
summary(pre_validation)

# See the distribution in the target variable. Its right skewed.
summary(pre_train$Item_Outlet_Sales)
qplot(x=Item_Outlet_Sales, data=pre_train)

summary( log2(pre_train$Item_Outlet_Sales) )


# Impute NA values in train.csv 
pre_train <- knnImputation(pre_train, k = 10)
pre_validation <- knnImputation(pre_validation, k = 10)
pre_test <- knnImputation(test_data, k=10)

# Count the number of factors in Outlet_Establishment_Year
count(pre_train, 'Outlet_Establishment_Year')
count(pre_test, 'Outlet_Establishment_Year')

# Replace Outlet_Establishment_Year with Number_of_Years
pre_train$Number_of_Years <- 2018 - pre_train$Outlet_Establishment_Year
pre_validation$Number_of_Years <- 2018 - pre_validation$Outlet_Establishment_Year
pre_test$Number_of_Years <- 2018 - pre_test$Outlet_Establishment_Year

summary(pre_test_data)

# Removing Item_Identifier, Outlet_Identifier and Outlet_Establishment_Year from train, test, validation data
train_data <- subset(x=pre_train, select=-c(Item_Identifier, Outlet_Identifier, Outlet_Establishment_Year))
validation_data <- subset(x=pre_validation, select=-c(Item_Identifier, Outlet_Identifier, Outlet_Establishment_Year))

pre_test_data <- subset(x=pre_test, select=-c(Item_Identifier, Outlet_Identifier, Outlet_Establishment_Year))

# Seperating numeric and categorical attributes
attr <- names(train_data)
num_attr <- subset(x=train_data, select=c(Item_Weight, Item_Visibility, Item_MRP))
cat_attr <- setdiff( names(train_data), num_attr)

# Summary of train_data
summary(train_data)
summary(validation_data)
summary(pre_test_data)

# Preparing the training data for logistic regression model creation
log_reg_train_data <- data.frame("Item_Weight" = train_data$Item_Weight,
                           "Item_Visibility" = train_data$Item_Visibility,
                           "Item_MRP" = train_data$Item_MRP,
                           "Number_of_Years" = train_data$Number_of_Years,
                           "y" = train_data$Item_Outlet_Sales)


log_reg_train_data <- log_reg_train_data %>%
                            mutate_each_( funs( log2(.+1) %>%
                                  as.vector), vars=c("Item_Weight", "Item_Visibility", "Item_MRP", "Number_of_Years") )
summary(log_reg_train_data)


##################### Logistic Regression Predictions on Test Data #########################
log_reg_test_data <- data.frame("Item_Weight" = test_data$Item_Weight,
                                 "Item_Visibility" = test_data$Item_Visibility,
                                 "Item_MRP" = test_data$Item_MRP,
                                 "Number_of_Years" = test_data$Number_of_Years)

log_reg_test_data <- log_reg_test_data %>%
                            mutate_each_( funs( log2(.+1) %>%
                                as.vector), vars=c("Item_Weight", "Item_Visibility", "Item_MRP", "Number_of_Years") )

summary(log_reg_test_data)
predict_test <- predict(log_reg, log_reg_test_data, type = "response")

export_df <- data.frame("Item_Identifier" = test_data$Item_Identifier,
                        "Outlet_Identifier" = test_data$Outlet_Identifier,
                        "Item_Outlet_Sales" = predict_test)
summary(export_df)
write.csv(export_df, file="logistic_regression.csv", na="")


# Creating the logistic regression model
log_reg <- glm(y~., data=log_reg_train_data)
vif(log_reg)

model_aic <- stepAIC(log_reg, direction = "both")
summary(model_aic)
summary(log_reg)


prob_log_reg_train <- predict(log_reg, type = "response")
summary(prob_log_reg_train)
df <- data.frame("y" = log_reg_train_data$y,
                        "y_predicted" = prob_log_reg_train)
Error<- (df$y - df$y_predicted)
df$Error <- data.frame("Error" = Error)
head(df)



summary(train_data)
summary(validation_data)
summary(test_data)






# Dummify the categorical attributes in train data
count(train_data, 'Item_Type')

Item_Fat_Content_DummyVars <- dummy(train_data$Item_Fat_Content)
train_data <- data.frame(subset(train_data, select = -c(Item_Fat_Content)), Item_Fat_Content_DummyVars)

Outlet_Size_DummyVars <- dummy(train_data$Outlet_Size)
train_data <- data.frame(subset(train_data, select = -c(Outlet_Size)), Outlet_Size_DummyVars)

Outlet_Type_DummyVars <- dummy(train_data$Outlet_Type)
train_data <- data.frame(subset(train_data, select = -c(Outlet_Type)), Outlet_Type_DummyVars)

Item_Type_DummyVars <- dummy(train_data$Item_Type)
train_data <- data.frame(subset(train_data, select = -c(Item_Type)), Item_Type_DummyVars)

Outlet_Location_Type_DummyVars <- dummy(train_data$Outlet_Location_Type)
train_data <- data.frame(subset(train_data, select = -c(Outlet_Location_Type)), Outlet_Location_Type_DummyVars)

Outlet_Establishment_Year_DummyVars <- dummy(train_data$Outlet_Establishment_Year)
train_data <- data.frame(subset(train_data, select = -c(Outlet_Establishment_Year)), Outlet_Establishment_Year_DummyVars)


# Dummify the categorical attributes in standardised validation data
Item_Fat_Content_DummyVars <- dummy(validation$Item_Fat_Content)
validation <- data.frame(subset(validation, select = -c(Item_Fat_Content)), Item_Fat_Content_DummyVars)

Outlet_Size_DummyVars <- dummy(validation$Outlet_Size)
validation <- data.frame(subset(validation, select = -c(Outlet_Size)), Outlet_Size_DummyVars)

Outlet_Type_DummyVars <- dummy(validation$Outlet_Type)
validation <- data.frame(subset(validation, select = -c(Outlet_Type)), Outlet_Type_DummyVars)

Item_Type_DummyVars <- dummy(validation$Item_Type)
validation <- data.frame(subset(validation, select = -c(Item_Type)), Item_Type_DummyVars)

Outlet_Location_Type_DummyVars <- dummy(validation$Outlet_Location_Type)
validation <- data.frame(subset(validation, select = -c(Outlet_Location_Type)), Outlet_Location_Type_DummyVars)

Outlet_Establishment_Year_DummyVars <- dummy(validation$Outlet_Establishment_Year)
validation <- data.frame(subset(validation, select = -c(Outlet_Establishment_Year)), Outlet_Establishment_Year_DummyVars)


# Dummify the categorical attributes in standardised test data
Item_Fat_Content_DummyVars <- dummy(pre_test_data$Item_Fat_Content)
pre_test_data <- data.frame(subset(pre_test_data, select = -c(Item_Fat_Content)), Item_Fat_Content_DummyVars)

Outlet_Size_DummyVars <- dummy(pre_test_data$Outlet_Size)
pre_test_data <- data.frame(subset(pre_test_data, select = -c(Outlet_Size)), Outlet_Size_DummyVars)

Outlet_Type_DummyVars <- dummy(pre_test_data$Outlet_Type)
pre_test_data <- data.frame(subset(pre_test_data, select = -c(Outlet_Type)), Outlet_Type_DummyVars)

Item_Type_DummyVars <- dummy(pre_test_data$Item_Type)
pre_test_data <- data.frame(subset(pre_test_data, select = -c(Item_Type)), Item_Type_DummyVars)

Outlet_Location_Type_DummyVars <- dummy(pre_test_data$Outlet_Location_Type)
pre_test_data <- data.frame(subset(pre_test_data, select = -c(Outlet_Location_Type)), Outlet_Location_Type_DummyVars)

Outlet_Establishment_Year_DummyVars <- dummy(pre_test_data$Outlet_Establishment_Year)
pre_test_data <- data.frame(subset(pre_test_data, select = -c(Outlet_Establishment_Year)), Outlet_Establishment_Year_DummyVars)


# correlation matrix
temp1 <- cor(std_train_data, use = "complete.obs") 
par(mfrow = c(1,1))
corrplot(temp1, type = "upper", method = "circle", order = "hclust", tl.col = "black", tl.srt = 90, tl.cex = 0.7)




### Random Forest
# Create the prediction model using Random Forest
model_rf <- randomForest(Item_Outlet_Sales~., data=std_train_data, keep.forest=TRUE, ntree=100, mtry = 4, importance = TRUE)
print(model_rf)
model_rf$importance

# Check model importance plot
rf_imp_attr <- data.frame(model_rf$importance)
rf_imp_attr <- data.frame( row.names(rf_imp_attr), rf_imp_attr[,1] )
colnames(rf_imp_attr) <- c("Attributes", "Importance")
rf_imp_attr <- rf_imp_attr[ order( rf_imp_attr$Importance, decreasing=TRUE), ]
varImpPlot(model_rf)


# Predict on train and check the accuracy
pred_train_model <- predict(model_rf, std_train_data[, setdiff(names(std_train_data), "Item_Outlet_Sales")], type = "response", norm.votes = TRUE)
cm_train <- table("actual"=std_train_data$Item_Outlet_Sales, "predicted"=pred_train_model)
#print(cm_train)
accu_train <- sum(diag(cm_train))/sum(cm_train)
accu_train

# Check the accuracy on validation
pred_validation_model <- predict(model_rf, std_validation_data[, setdiff(names(std_validation_data), "Item_Outlet_Sales")], type = "response", norm.votes = TRUE)
cm_validation <- table("actual"= std_validation_data$Item_Outlet_Sales, "predicted"= pred_validation_model)
#print(cm_train)
accu_validation <- sum(diag(cm_validation))/sum(cm_validation)
accu_validation


# Take the important attributes and rebuild the Random Forest model
top_Imp_Attr <- as.character(rf_imp_attr$Attributes[1:15])
set.seed(365)
model_imp <- randomForest(Item_Outlet_Sales~., data = std_train_data[,c(top_Imp_Attr,"Item_Outlet_Sales")], keep.forest=TRUE, ntree=100)
print(model_imp)
model_imp$importance
varImpPlot(model_imp)

# Find the best.m
mtry <- tuneRF(std_train_data[,!colnames(std_train_data)%in%"Item_Outlet_Sales"],std_train_data$Item_Outlet_Sales,ntreeTry = 100,stepFactor = 1.5,improve = 0.01,trace = TRUE,plot = TRUE)
best.m <- mtry[mtry[,2]==min(mtry[,2]),1]
print(best.m)

# Check with a different seed value
set.seed(265)
rf <- randomForest(Item_Outlet_Sales~., data=std_train_data, mtry=best.m, importance=TRUE, ntree=100)
print(rf)
#rf$importance
round(rf$importance,2)





##Predict on train data
pred_train <- predict(rf, std_train_data[, setdiff(names(std_train_data), "Item_Outlet_Sales")], type = "response", norm.votes = TRUE)

##Build Confution matrix
cm_train <- table("actual"=std_train_data$Item_Outlet_Sales, "Predicted"=pred_train)
#cm_train
accu_train_rf <- sum(diag(cm_train))/sum(cm_train)
accu_train_rf
rm(pred_train,cm_train)

##Predict on validation data
pred_val <- predict(rf, std_validation_data[, setdiff(names(std_validation_data), "Item_Outlet_Sales")], type = "response", norm.votes = TRUE)

##build confusion matrix
cm_val <- table("actual"=std_validation_data$Item_Outlet_Sales, "predict"=pred_val)
accu_val_rf <- sum(diag(cm_val))/sum(cm_val)
accu_val_rf




##Predict on test data
pred_test <- predict(rf, std_test_data[, setdiff(names(std_test_data), "Item_Outlet_Sales")], type = "response", norm.votes = TRUE)

index_value <- data.frame(prediction = pred_test)
write.csv(index_value,"1submission.csv")

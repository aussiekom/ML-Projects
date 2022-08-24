# import all the libraries that we will need during the project
library(tidyverse) # metapackage of all tidyverse packages
library(mlbench) # Data modelling
library(e1071) # Naive Bayes
library(caret) # Data preprocessing, modelling
library(Amelia) # Missing plot
library(DataExplorer) # Visualization
library(MASS) # Functions and datasets
library(klaR) # Classification and Visualization
library(kernlab) # For ML methods
library(randomForest)

# Hide warnings
options(warn=-1) 
list.files(path = "../input")

# Upload dataset
dataset_path <- 'parkinsons.data'
data <- read.csv(dataset_path)
head(data)
# Check dimensions
dim(data)
# Check datatypes
sapply(data, class)

#----- The output variable is of data type integer, which needs to be converted to factor, 
# considering this is a classification problem.
#----- Rest all variables are of numeric data type.

# Check class distribution
y <- data$status
cbind(freq=table(y), percentage=prop.table(table(y))*100)


#---- The dataset is slightly imbalanced, with positive samples thrice as much as negative samples.
#---- We will check how it impacts performance, and perform rebalancing techniques if necessary.

# Check summary statistics
summary(data)

#---- Multiple variables have varied min/max value ranges, data scaling might help
sapply(data, sd)

# Check NA values
sum(is.na(data))
# There is no missing values


# Data visualization
# Histograms of each attribute
for(i in 2:24) {
  print(ggplot(data, aes(x=data[,i])) + geom_histogram(bins=30, aes(y=..density..), colour="black", fill="white")
        + geom_density(alpha=.2, fill="#FF6666"))
}

# Check missing values
missmap(data, col=c('black','grey'),legend=FALSE)

# ------ From above exploratory data analysis, we can conclude that:
# ------ 1. Status is the dependent variable, which is of class integer
# ------ 2. All independent variables are of class numeric, expect for name which is of class character
# ------ 3. There are three times as much positive samples, as there are negative samples
# ------ 4. All independent variables have acceptable variance
# ------ 5. Multiple independent variables are highly correlated with each other
# ------ 6. There are no null values in the dataset


# Data Preparation
#---- We know that there are no missing values in our dataset.
#---- We will check variables with very low variance and consider removing them.

# Remove columns with near zero variance
# Calculate preprocessing parameters from the dataset
varParams <- preProcess(data[, -17], method=c("nzv"))
# Summarize preprocess parameters
print(varParams)

#---- Since all variables were ignored, we can conclude that all variables have acceptable variance.

# Feature selection
# Dropping name column, as it holds no weight as a predictor
data <- data[,2:24]
# Check if name column has been dropped
head(data, 2)

# Find highly correlated independent variables
print(findCorrelation(cor(data), cutoff=0.9))
# Remove highly correlated independent variables
data <- data[, -c(findCorrelation(cor(data), cutoff=0.9))]
# Dataset without high correlation between independent variables
head(data, 5)

#---- We saw in the data summarization step, that multiple predictors were highly correlated.
#---- Above function finds those predictors and we promptly remove them.


# Data transforms
# For classification, the dependent variable should be of class factor
data$status = as.factor(data$status)
# Recheck classes of all variables
sapply(data, class)


# Scale dataset
# Calculate preprocessing parameters from the dataset
normParams <- preProcess(data[, -17], method=c("range"))
# Summarize preprocess parameters
print(normParams)
# Transform the dataset using above parameters
data[, -17] <- predict(normParams, data[, -17])
summary(data)

#---- During our EDA, we had found that all variables had a different scale.
#---- Rescaling them to a common scale might increase performance for instance based and weight based algorithms.

# Modelling
# Prepare resampling method
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"

# Logistic Regression
set.seed(7)
fit.lr <- train(status ~ ., data=data, method="glm", metric=metric, trControl=trainControl)
# CART
set.seed(7)
fit.cart <- train(status ~ ., data=data, method="rpart", metric=metric, trControl=trainControl)
# LDA
set.seed(7)
fit.lda <- train(status ~ ., data=data, method="lda", metric=metric, trControl=trainControl)
# Naive Bayes
set.seed(7)
fit.nb <- train(status ~ ., data=data, method="nb", metric=metric, trControl=trainControl)
# SVM
set.seed(7)
fit.svm <- train(status ~ ., data=data, method="svmRadial", metric=metric, trControl=trainControl)
# KNN
set.seed(7)
fit.knn <- train(status ~ ., data=data, method="knn", metric=metric, trControl=trainControl)
# Random Forest
set.seed(7)
fit.rf <- train(status ~ ., data=data, method="rf", metric=metric, trControl=trainControl)

# Collect resamples
results <- resamples(list(LR=fit.lr, CART=fit.cart, NB=fit.nb, SVM=fit.svm, KNN=fit.knn, RF=fit.rf))
summary(results)
dotplot(results)












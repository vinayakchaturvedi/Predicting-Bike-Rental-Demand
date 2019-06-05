#Clear the environment
rm(list=ls())

#set the working directory
setwd(dir = "C:/Users/vinayak/Desktop/Bike Renting Project")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees',"createDataPartition","usdm","randomForest","e1071")

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Load the dataset
dataset = read.csv("day.csv")

#Check the structure of the dataset
str(dataset)

##################################Missing value analysis################################################
sum(is.na(dataset))
#There are no missing values in the dataset so no need to perform missing value analysis.

#Univariate Analysis and Variable Consolidation
dataset$season = as.factor(dataset$season)
dataset$yr = as.factor(dataset$yr)
dataset$mnth = as.factor(dataset$mnth)
dataset$holiday = as.factor(dataset$holiday)
dataset$weekday = as.factor(dataset$weekday)
dataset$workingday = as.factor(dataset$workingday)
dataset$weathersit = as.factor(dataset$weathersit)

#Remove the variable that are not useful for the analysis
#1. Instant there is no need to add it as it only explains the row number
#2. Dteday there is no need to add date as year and month are already present in 2 different columns and date 
#   does not have large impact on the result
#3. Casual and Registered: Ideally we should not use these 2 variable because cnt = casual and Registered
#   and we need to calculate the bike count on the basis of environmental and seasonal settings.

c = c(-1,-2,-14,-15)
dataset = dataset[c]

##################################Outlier Analysis################################################

numeric_index = sapply(dataset,is.numeric) #selecting only numeric
numeric_data = dataset[,numeric_index]

factor_index = sapply(dataset,is.factor)  #selecting only factor
factor_data = dataset[,factor_index]


cnames = colnames(numeric_data)
# 
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(dataset))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="cnt")+
           ggtitle(paste("Box plot of cnt for",cnames[i])))
}
#
## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,ncol=3)

for(i in cnames){
  val = dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  #print(length(val))
  dataset[,i][dataset[,i] %in% val] = NA
}
sum(is.na(dataset))

#Impute NA using KNN impute
dataset = knnImputation(dataset, k = 3)

##################################Feature Selection################################################
## Correlation Plot 
corrgram(dataset[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


#As we can see in the plot that that temp and atemp are very high +vely correlated then we can drop any 1 
#So here I am dropping atemp.
dataset = dataset[,-9]

##################################Feature scaling################################################
# In the dataset there are only 3 numeric variables temp, hum, windspeed and the value of these variable 
# are in the range of 0 and 1, so no need to apply feature scaling.

###################################Model Development#######################################
#Clean the environment
rmExcept("dataset")


#Divide data into train and test set
set.seed(1234)
train_index = sample(1:nrow(dataset), 0.8 * nrow(dataset))
train = dataset[train_index,]
test = dataset[-train_index,]

#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}


#rpart for regression
fit = rpart(cnt ~ ., data = train, method = "anova")
predictions_DT = predict(fit, test[,-11])     #Predict for new test cases
MAPE(test[,11], predictions_DT)

#Error Rate: 21.98337
#Accuracy: 78.01663

#run regression model
lm_model = lm(cnt ~., data = train)
predictions_LR = predict(lm_model, test[,-11])    #Predict
MAPE(test[,11], predictions_LR)                   #Calculate MAPE

#Error Rate: 20.47
#acuracy: 79.53%

# Fitting Random Forest Regression to the dataset
set.seed(1234)
regressor = randomForest(x = train[-11], y = train$cnt,ntree = 500)
predictions_regressor = predict(regressor, test[,-11])
MAPE(test[,11], predictions_regressor)

#Error Rate: 19.58
#acuracy: 80.42%

# Fitting SVR to the dataset
regressor = svm(formula = cnt ~ .,data = train,type = 'eps-regression',kernel = 'radial')
predictions_regressor = predict(regressor, test[,-11])
MAPE(test[,11], predictions_regressor)

#Error Rate: 16.44
#acuracy: 83.56%

#Pass sample input into the model
sample = data.frame(season=1,	yr=0,	mnth=1,	holiday=0,	weekday=1,	workingday=1,	weathersit=1,	
                    temp=0.160833,	hum=0.492917, windspeed=0.223267, cnt=1321)
newtest = rbind(test, sample)
predict(regressor, newtest[148,-11])


#You might see some slight differences in the accuracy as i applied outlier anlysis quite a few times
#to remove outliers.
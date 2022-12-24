###Data preparation

# load raw data
# https://drive.google.com/file/d/1Iee3Xu-Auy1ipEH3EneZVlNZrgzdrNw2/view?usp=share_link
system("gdown --id 1Iee3Xu-Auy1ipEH3EneZVlNZrgzdrNw2")
historic_property_data_raw <- read.csv("historic_property_data.csv")

#https://drive.google.com/file/d/11P7dzl9HgvlIGpUqfBakwoxHnAnwmlSO/view?usp=sharing
system("gdown --id 11P7dzl9HgvlIGpUqfBakwoxHnAnwmlSO")
predict_raw <- read.csv("predict_property_data.csv")

historic_property_data <- historic_property_data_raw
predict <- predict_raw

# load packages
library(dplyr)
install.packages('fastDummies')
library(fastDummies)
install.packages('caret')
library(caret)
library(magrittr)
install.packages('randomForest')
library(randomForest)

# convert to character
historic_property_data$meta_class                  <- as.character(historic_property_data$meta_class)
historic_property_data$meta_town_code              <- as.character(historic_property_data$meta_town_code)
historic_property_data$char_ext_wall               <- as.character(historic_property_data$char_ext_wall)
historic_property_data$char_roof_cnst              <- as.character(historic_property_data$char_roof_cnst)
historic_property_data$char_bsmt                   <- as.character(historic_property_data$char_bsmt)
historic_property_data$char_bsmt_fin               <- as.character(historic_property_data$char_bsmt_fin)
historic_property_data$char_heat                   <- as.character(historic_property_data$char_heat)
historic_property_data$char_oheat                  <- as.character(historic_property_data$char_oheat)
historic_property_data$char_air                    <- as.character(historic_property_data$char_air)
historic_property_data$char_attic_type             <- as.character(historic_property_data$char_attic_type)
historic_property_data$char_tp_plan                <- as.character(historic_property_data$char_tp_plan)
historic_property_data$char_cnst_qlty              <- as.character(historic_property_data$char_cnst_qlty)
historic_property_data$char_site                   <- as.character(historic_property_data$char_site)
historic_property_data$char_gar1_size              <- as.character(historic_property_data$char_gar1_size)
historic_property_data$char_gar1_cnst              <- as.character(historic_property_data$char_gar1_cnst)
historic_property_data$char_gar1_att               <- as.character(historic_property_data$char_gar1_att)
historic_property_data$char_gar1_area              <- as.character(historic_property_data$char_gar1_area)
historic_property_data$char_repair_cnd             <- as.character(historic_property_data$char_repair_cnd)
historic_property_data$char_use                    <- as.character(historic_property_data$char_use)
historic_property_data$char_type_resd              <- as.character(historic_property_data$char_type_resd)
historic_property_data$geo_fips                    <- as.character(historic_property_data$geo_fips)
historic_property_data$geo_ohare_noise             <- as.character(historic_property_data$geo_ohare_noise)
historic_property_data$geo_floodplain              <- as.character(historic_property_data$geo_floodplain)
historic_property_data$geo_fs_flood_risk_direction <- as.character(historic_property_data$geo_fs_flood_risk_direction)
historic_property_data$geo_withinmr100             <- as.character(historic_property_data$geo_withinmr100)
historic_property_data$geo_withinmr101300          <- as.character(historic_property_data$geo_withinmr101300)
historic_property_data$ind_large_home              <- as.character(historic_property_data$ind_large_home)
historic_property_data$ind_garage                  <- as.character(historic_property_data$ind_garage)
historic_property_data$ind_arms_length             <- as.character(historic_property_data$ind_arms_length)

predict$meta_class                  <- as.character(predict$meta_class)
predict$meta_town_code              <- as.character(predict$meta_town_code)
predict$char_ext_wall               <- as.character(predict$char_ext_wall)
predict$char_roof_cnst              <- as.character(predict$char_roof_cnst)
predict$char_bsmt                   <- as.character(predict$char_bsmt)
predict$char_bsmt_fin               <- as.character(predict$char_bsmt_fin)
predict$char_heat                   <- as.character(predict$char_heat)
predict$char_oheat                  <- as.character(predict$char_oheat)
predict$char_air                    <- as.character(predict$char_air)
predict$char_attic_type             <- as.character(predict$char_attic_type)
predict$char_tp_plan                <- as.character(predict$char_tp_plan)
predict$char_cnst_qlty              <- as.character(predict$char_cnst_qlty)
predict$char_site                   <- as.character(predict$char_site)
predict$char_gar1_size              <- as.character(predict$char_gar1_size)
predict$char_gar1_cnst              <- as.character(predict$char_gar1_cnst)
predict$char_gar1_att               <- as.character(predict$char_gar1_att)
predict$char_gar1_area              <- as.character(predict$char_gar1_area)
predict$char_repair_cnd             <- as.character(predict$char_repair_cnd)
predict$char_use                    <- as.character(predict$char_use)
predict$char_type_resd              <- as.character(predict$char_type_resd)
predict$geo_fips                    <- as.character(predict$geo_fips)
predict$geo_ohare_noise             <- as.character(predict$geo_ohare_noise)
predict$geo_floodplain              <- as.character(predict$geo_floodplain)
predict$geo_fs_flood_risk_direction <- as.character(predict$geo_fs_flood_risk_direction)
predict$geo_withinmr100             <- as.character(predict$geo_withinmr100)
predict$geo_withinmr101300          <- as.character(predict$geo_withinmr101300)
predict$ind_large_home              <- as.character(predict$ind_large_home)
predict$ind_garage                  <- as.character(predict$ind_garage)
predict$ind_arms_length             <- as.character(predict$ind_arms_length)

idx_feat_cat_mask <- grepl('character', sapply(historic_property_data, class))     # Obtain feature index masks

idx_feat_cat_tmp <- colnames(predict)[idx_feat_cat_mask]  # Obtain feature indices
predict_cat <- predict[, idx_feat_cat_tmp]             # Subset master df with feature indices
dim(predict_cat)

idx_feat_cat_tmp <- colnames(historic_property_data)[idx_feat_cat_mask]    # Obtain feature indices
train_cat <- historic_property_data[, idx_feat_cat_tmp]                 # Subset master df with feature indices
dim(train_cat)
str(train_cat)

na_summary <- cbind( as.data.frame(colnames(train_cat)),
                     as.data.frame(colMeans(is.na(train_cat))))   # Combine feature names & feature NA counts
colnames(na_summary)[1] <- "feature"                                 # Rename column
colnames(na_summary)[2] <- "na_cnt"                                  # Rename column
na_summary <- na_summary[na_summary$na_cnt>0, ]                # Subset feature names (rows) with NAs
na_summary <- na_summary[order(-na_summary$na_cnt),]           # Sort by 'na_cnt' (desc)
na_summary  

idx_feat_excl <- c( 'meta_town_code',                     # exclude geo features 
                    'geo_property_city',
                    'geo_fips',
                    'geo_municipality',
                    'geo_school_elem_district',
                    'geo_school_hs_district')

idx_feat_sub_mask_tmp <- setdiff(colnames(train_cat), idx_feat_excl)     # Obtain feature indices
train_cat <- train_cat[, idx_feat_sub_mask_tmp]                       # Subset train_cat with feature indices
dim(train_cat)

idx_feat_sub_mask_tmp <- setdiff(colnames(predict_cat), idx_feat_excl)   # Obtain feature indices
predict_cat <- predict_cat[, idx_feat_sub_mask_tmp]                   # Subset predict_cat with feature indices
dim(predict_cat)

idx_feat_num_mask <- !idx_feat_cat_mask                    # obtain numeric features           

idx_feat_num_tmp <- colnames(historic_property_data)[idx_feat_num_mask]    # Obtain feature indices 
train_num <- historic_property_data[, idx_feat_num_tmp]                  # Subset train_num with feature indices 
dim(train_num)                                                      

idx_feat_num_tmp <- colnames(predict)[idx_feat_num_mask]            # Obtain feature indices  
predict_num <- predict[, idx_feat_num_tmp]                        # Subset predict_num with feature indices
dim(predict_num)                                                    

# numeric features na value exploration 
na_summary <- cbind( as.data.frame(colnames(train_num)),           
                     as.data.frame(colMeans(is.na(train_num))))  
colnames(na_summary)[1] <- "variable"                                 
colnames(na_summary)[2] <- "na_percentage"                                  
na_summary <- na_summary[na_summary$na_percentage>0, ]                
na_summary <- na_summary[order(-na_summary$na_percentage),]           
na_summary

# drop features that contain over 35% NA values
thresh = 0.35

train_cat <- train_cat[, which(colMeans(is.na(train_cat)) <= thresh)]      # train data categorical features with more than 35% NAs
train_num <- train_num[, which(colMeans(is.na(train_num)) <= thresh)]      # train data numeric features with more than 35% NAs
dim(train_cat)
dim(train_num)

predict_cat <- predict_cat[, which(colMeans(is.na(predict_cat)) <= thresh)]    # predict data categorical features with more than 35% NAs    
predict_num <- predict_num[, which(colMeans(is.na(predict_num)) <= thresh)]    # predict data numeric features with more than 35% NAs
dim(predict_cat)
dim(predict_num)

# fill NAs with mode for categorical features
Na_mode <- function(sample_series) {
  unique_samples <- unique(sample_series)
  unique_samples[which.max(tabulate(match(sample_series, unique_samples)))]
}

# fill NAs with mean for numerical features
train_cat %>% mutate_all(~ifelse(is.na(.x), Na_mode(.x), .x))
predict_cat %>% mutate_all(~ifelse(is.na(.x), Na_mode(.x), .x))

# combine catgorical features and numeric features
historic_property_data <- cbind(train_cat, train_num)       
dim(historic_property_data)

predict <- cbind(predict_cat, predict_num) 
dim(predict)

# replace redundant categories
historic_property_data$char_type_resd[historic_property_data$char_type_resd == '6' |historic_property_data$char_type_resd == '7'] <- '5'     # Replace class values
historic_property_data$char_type_resd[historic_property_data$char_type_resd == '8' |historic_property_data$char_type_resd == '9'] <- '5'     # Replace class values
unique(historic_property_data$char_type_resd)

predict$char_type_resd[predict$char_type_resd == '6' |predict$char_type_resd == '7'] <- '5'   # Replace class values
predict$char_type_resd[predict$char_type_resd == '8' |predict$char_type_resd == '9'] <- '5'   # Replace class values
unique(predict$char_type_resd)

# ensure historic data and predict data had the same features
historic_property_data <- historic_property_data[!(historic_property_data$geo_property_zip == '60661' | historic_property_data$geo_property_zip == '60658'), ]
dim(historic_property_data)

# separate categorical and numeric features
idx_feat_cat_mask <- grepl('character', sapply(historic_property_data, class))
idx_feat_num_mask <- !idx_feat_cat_mask
idx_feat_cat <- colnames(historic_property_data)[idx_feat_cat_mask]

idx_feat_num_tmp <- colnames(historic_property_data)[idx_feat_num_mask]
train_num <- historic_property_data[, idx_feat_num_tmp]
train_cat <- historic_property_data[, idx_feat_cat]

idx_feat_num_tmp <- colnames(predict)[idx_feat_num_mask]
predict_num <- predict[, idx_feat_num_tmp]
predict_cat <- predict[, idx_feat_cat]

# create dummy variables for categorical features
train_cat_enc <- dummy_cols(.data=train_cat, remove_first_dummy=T, remove_selected_columns=T)  
dim(train_cat_enc)

predict_cat_enc <- dummy_cols(.data=predict_cat, remove_first_dummy=T, remove_selected_columns=T)  
dim(predict_cat_enc)

# create dummy variables for categorical features
train_cat_enc <- dummy_cols(.data=train_cat, remove_first_dummy=T, remove_selected_columns=T)  
dim(train_cat_enc)

predict_cat_enc <- dummy_cols(.data=predict_cat, remove_first_dummy=T, remove_selected_columns=T)  
dim(predict_cat_enc)

# Train/test split
predict_idx <- as.data.frame(predict$pid)                                      
colnames(predict_idx) <- 'pid'                                                           
predict <- predict[, setdiff(colnames(predict), c('pid'))]   
dim(predict)

# Train/test split
set.seed(917)     

train_sample <- sample(c(1:dim(historic_property_data)[1]), dim(historic_property_data)[1]*0.6)  
df_train <- historic_property_data[train_sample, ]                                        
df_test <- historic_property_data[-train_sample, ]                                        

train_y <- as.data.frame(df_train$sale_price)                        
colnames(train_y) <- "sale_price"                                    
train_X <- df_train[, setdiff(colnames(df_train), c('sale_price'))]  
dim(train_X)

test_y <- as.data.frame(df_test$sale_price)                          
colnames(test_y) <- "sale_price"                                    
test_X <- df_test[, setdiff(colnames(df_test), c('sale_price'))]    
dim(test_X)

##Linear Regression

#Linear regression with all features

# fit the model with all predictors  
lm_full <- lm(sale_price ~ ., data = df_train)

# summary table 
summary(lm_full) 

#make predictions using lm

# make predictions on the test set
lm_full_pred <- predict(lm_full, df_test)
# MSE in the test set 
mean((df_test$sale_price-lm_full_pred)^2)

##Random Forest

#Feature selection

# Drop column names of the dataframe which starts with
df_train_rf <- select(df_train,-starts_with("geo_property_zip"))
df_train_rf <- select(df_train_rf,-starts_with("char_cnst_qlty"))

# Drop column names of the dataframe which starts with
df_test_rf <- select(df_test,-starts_with("geo_property_zip"))
df_test_rf <- select(df_test_rf,-starts_with("char_cnst_qlty"))

#fit the rf model

# install.packages('randomForest')
library(randomForest)

rf1 <- randomForest(sale_price ~ ., data = df_train_rf, ntree=50, norm.votes=FALSE, na.action = na.roughfix)
rf2 <- randomForest(sale_price ~ ., data = df_train_rf, ntree=50, norm.votes=FALSE, na.action = na.roughfix)
rf3 <- randomForest(sale_price ~ ., data = df_train_rf, ntree=50, norm.votes=FALSE, na.action = na.roughfix)
rf4 <- randomForest(sale_price ~ ., data = df_train_rf, ntree=50, norm.votes=FALSE, na.action = na.roughfix)
rf5 <- randomForest(sale_price ~ ., data = df_train_rf, ntree=50, norm.votes=FALSE, na.action = na.roughfix)
rf.all <- combine(rf1, rf2, rf3, rf4, rf5)

# make predictions for records in the test set
rf.all.predict <-predict(rf.all, df_test_rf)

rf2.all.predict <-predict(rf2, df_test_rf)

plot(df_test_rf$sale_price, rf.all.predict, main = 'testing set',
     xlab = 'Sale Price', ylab = 'Predict')
abline(1, 1)

# Mean Squared Error
mean((df_test_rf$sale_price- rf.all.predict)**2)
# RMSE
sqrt(mean((df_test_rf$sale_price- rf.all.predict)**2))

#Use predict dataset

predict_rf <- predict

# make predictions for records in the test set
rf.predict <-predict(rf.all, predict_rf)

rf.predict.frame <- as.data.frame(rf.predict)
rf.predict.frame

write.csv(rf.predict.frame,"predict_rf.csv", row.names = FALSE)

summary(predict_rf[1])

library(ggplot2)
library(dplyr)

ggplot(data = rf.predict.frame,aes(x=rf.predict)) +
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8)

#variable importance

# feature selection
#importance()
importance_label <- data.frame(importance(rf.all), check.names = FALSE)
head(importance_label)

# top 30 variable importance plot
varImpPlot(rf.all, n.var = min(30, nrow(rf.all$importance)),
           main = 'Top 30 - variable importance')
# %IncMSE: increase in mean squared error
# IncNodePurity: increase in node purity

# sort the variables according to "IncNodePurity" 
importance_label  <- importance_label[order(importance_label$IncNodePurity, decreasing = TRUE), ]
head(importance_label)

# summary table
# write.table(importance_label, 'importance_label.txt', sep = '\t', col.names = NA, quote = FALSE)

##XGBoost

#Feature selection

# Drop column names of the dataframe which starts with
df_train_rf <- select(df_train,-starts_with("geo_property_zip"))
df_train_xgb <- select(df_train_rf,-starts_with("char_cnst_qlty"))

# Drop column names of the dataframe which starts with
df_test_rf <- select(df_test,-starts_with("geo_property_zip"))
df_test_xgb <- select(df_test_rf,-starts_with("char_cnst_qlty"))

#fit the xgb model

# install.packages('xgboost')
library(xgboost)

train_y_xgb <- as.data.frame(df_train_xgb$sale_price)                        
colnames(train_y_xgb) <- "sale_price"                                    
train_X_xgb <- df_train_xgb[, setdiff(colnames(df_train_xgb), c('sale_price'))]  
dim(train_X_xgb)

test_y_xgb <- as.data.frame(df_test_xgb$sale_price)                          
colnames(test_y_xgb) <- "sale_price"                                    
test_X_xgb <- df_test[, setdiff(colnames(df_test_xgb), c('sale_price'))]    
dim(test_X_xgb)

m1_xgb <-
  xgboost(
    data = data.matrix(train_X_xgb),
    label=df_train_xgb$sale_price,
    nrounds = 1000,
    objective = "reg:squarederror",
    early_stopping_rounds = 3,
    max_depth = 5,
    eta = .24
  )   

pred_xgb <- predict(m1_xgb, data.matrix(test_X_xgb))

yhat <- pred_xgb
y <- df_test_xgb[, 1]
postResample(yhat, y)

r <- y - yhat
plot(r, ylab = "residuals")

plot(y,
     yhat,
     xlab = "actual",
     ylab = "predicted")
abline(lm(yhat ~ y))

# install.packages('DiagrammeR')
library(DiagrammeR)

#plot first 3 trees of model
xgb.plot.tree(model = m1_xgb, trees = 0:2)

importance_matrix <- xgb.importance(model = m1_xgb)
xgb.plot.importance(importance_matrix, xlab = "Feature Importance")

# Mean Squared Error
mean((df_test_xgb$sale_price-pred_xgb)**2)
# RMSE
sqrt(mean((df_test_xgb$sale_price- pred_xgb)**2))

#Best parameter(optional)

#create hyperparameter grid
hyper_grid <- expand.grid(max_depth = seq(3, 6, 1), eta = seq(.2, .35, .01))  

for (j in 1:nrow(hyper_grid)) {
  set.seed(123)
  m_xgb_untuned <- xgb.cv(
    data = data.matrix(train_X_xgb),
    label=df_train_xgb$sale_price,
    nrounds = 1000,
    objective = "reg:squarederror",
    early_stopping_rounds = 3,
    nfold = 5,
    max_depth = hyper_grid$max_depth[j],
    eta = hyper_grid$eta[j]
  )
  
  xgb_train_rmse[j] <- m_xgb_untuned$evaluation_log$train_rmse_mean[m_xgb_untuned$best_iteration]
  xgb_test_rmse[j] <- m_xgb_untuned$evaluation_log$test_rmse_mean[m_xgb_untuned$best_iteration]
  
  cat(j, "\n")
}

#ideal hyperparamters
hyper_grid[which.min(xgb_test_rmse), ]

#max_depth  eta
#5         0.24 

##Neural Network

#Install Package

install.packages('neuralnet')
library(neuralnet)

#Normalization

# Scale the data to the interval between zero and one
# Normalize the numerical predictors and the outcome variable in the Training Data
df_train_val <- preProcess(df_train, method="range")    
df_norm_train <- predict(df_train_val, df_train) 

# Normalize the numerical predictors and the outcome variable in the Test Data
df_test_val <- preProcess(df_test, method="range")    
df_norm_test <- predict(df_test_val, df_test)

#Fit Model

#Fit the selected variables in the model
nn1 <- neuralnet(sale_price ~ meta_class_204+meta_class_205+meta_class_208+meta_class_209+meta_class_211+meta_class_212+meta_class_278+meta_deed_type_T+meta_deed_type_W+char_ext_wall_3+char_ext_wall_4+char_roof_cnst_2+char_roof_cnst_3+char_roof_cnst_5+char_roof_cnst_6+char_bsmt_2+char_bsmt_3+char_bsmt_fin_3+char_heat_4+char_air_2+char_site_2+char_gar1_size_7+char_gar1_cnst_2+char_gar1_att_2+char_gar1_area_2+char_repair_cnd_3+char_type_resd_3+char_type_resd_4+geo_ohare_noise_1+geo_fs_flood_risk_direction_1+geo_withinmr100_1+geo_withinmr101300_1+ind_large_home_TRUE+ind_garage_TRUE+ind_arms_length_TRUE+sale_price+meta_nbhd+meta_certified_est_bldg+meta_certified_est_land+char_hd_sf+char_age+char_rooms+char_beds+char_fbath+char_hbath+char_ot_impr+char_bldg_sf+geo_tract_pop+geo_white_perc+geo_black_perc+geo_his_perc+geo_other_perc+econ_tax_rate+econ_midincome, 
                 data = df_norm_train, linear.output = TRUE, hidden = 2)

#Make Prediction
predict.nn1 <- compute(nn1, df_norm_test)
head(predict.nn1$net.result)
#Check the RMSE
sqrt(mean((df_norm_test$sale_price-predict.nn1$net.result)^2))

##LASSO REGRESSION

#Data Visualisation


options(repr.plot.width=10, repr.plot.height=8)
plot(x=historic_property_data$sale_price, y=historic_property_data$meta_certified_est_bldg, pch=18, cex=2, col="orange",cex.main=1.5, cex.lab = 1.5,cex.axis=1, xlab="Meta Certified East Buiding", ylab="Sale Price", main="Scatter plot showing Sale Price vs Meta_certified_est_bldg") 

install.packages('glmnet')
library(glmnet)

test_sample <- sample(c(1:dim(df_test)[1]), dim(df_test)[1]*1) 

# convert a data frame of predictors to a matrix and create dummy variables for character variables 

x <- model.matrix(sale_price~.,historic_property_data)[,-1]

y.test <- y[test_sample] # outcome in the test set

# fit a lasso regression model 

fit<- glmnet(x[train_sample,],y[train_sample],alpha=1)

# Return a medium lambda value 
lambda.medium <-fit$lambda[20]
lambda.medium

# lasso regression coefficients  
coef.lambda.medium <- predict(fit,s=lambda.medium,type="coefficients")[1:20,]
coef.lambda.medium

# non-zero coefficient estimates  
coef.lambda.medium[coef.lambda.medium!=0]

# make predictions for records the test set 
pred.lambda.medium <- predict(fit,s=lambda.medium,newx=x[test_sample,])
head(pred.lambda.medium)


# MSE in the test set 
mean((y.test-pred.lambda.medium)^2)

#RMSE 
sqrt(mean((y.test-pred.lambda.medium)^2))

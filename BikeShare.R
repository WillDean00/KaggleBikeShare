install.packages(tidy) 
install.packages("DataExplorer")
library(tidyverse)
library(tidymodels)
library(vroom)

train <- vroom("train.csv")
test <- vroom("test.csv")
vroom(train)
intro <- plot_intro(train)
corr <- plot_correlation(train)
hist <- plot_histogram(train)
bar <- plot_bar(train)
miss <- plot_missing(train)
(corr + hist) / (intro + bar)

library(DataExplorer)
library(patchwork)

glimpse(train)


my_linear_model <- linear_reg() %>% 
set_engine("lm") %>% 
set_mode("regression") %>% 
fit(formula = count ~ temp + humidity, data=train)


bike_predictions <- predict(my_linear_model,
                            new_data=test) 
bike_predictions

kaggle_submission <- bike_predictions %>%
bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write out the file9
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")




#poisson regression
install.packages("poissonreg")
library(poissonreg)

my_pois_model <- poisson_reg() %>% 
  set_engine("glm") %>% 
  set_mode("regression") %>%
fit(formula=count ~ temp + humidity, data=train)
bike_predictions <- predict(my_pois_model,
                            new_data=test) 
bike_predictions ## Look at the output


pois_kaggle_submission <- bike_predictions %>%
bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

vroom_write(x=pois_kaggle_submission, file="./PoissonPreds.csv", delim = ",")



#Data Wrangling
new_train <- train %>% 
  select(-casual, -registered) %>% 
  mutate(count = log(count)) 

glimpse(new_train)
view(train)
glimpse(my_recipe)

my_recipe <- recipe(count ~ ., data = new_train) %>% 
  step_mutate(weather =ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3)) %>% 
  step_time(datetime, features = c("hour")) %>% 
  step_mutate(season = factor(season, levels = 1:4, labels = c("spring", "summer", "fall" , "winter"))) %>% 
  step_rm(datetime, temp, holiday, workingday) %>% 
  step_mutate(windspeed = ifelse(windspeed == 0, 0, 1))


prep_recipe <- prep(my_recipe)
baked <- bake(prep_recipe, new_data = new_train)
view(baked)

lin_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(lin_model) %>% 
  fit(data = new_train)
new_train

lin_preds <- predict(bike_workflow, new_data = test)
exp_lin_preds <- exp(lin_preds)


workflow_kaggle_submission <- exp_lin_preds %>%
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

vroom_write(x=workflow_kaggle_submission, file=".WorkflowPreds.csv", delim = ",")

#Penalized Regression
my_recipe_2 <- recipe(count ~ ., data = new_train) %>% 
  step_mutate(weather =ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3)) %>% 
  step_time(datetime, features = c("hour")) %>% 
  step_mutate(datetime_hour = as.factor(datetime_hour))%>%
  step_mutate(season = factor(season, levels = 1:4, labels = c("spring", "summer", "fall" , "winter"))) %>% 
  step_rm(datetime, temp, holiday, workingday) %>% 
  step_mutate(windspeed = ifelse(windspeed == 0, 0, 1))%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 
  
prep_recipe <- prep(my_recipe_2)
  baked <- bake(prep_recipe, new_data = new_train)
  view(baked)
  
  preg_model <- linear_reg(penalty=0, mixture=.69) %>% 
    set_engine("glmnet")
  
  preg_wf <- workflow() %>%
  add_recipe(my_recipe_2) %>%
  add_model(preg_model) %>%
  fit(data=new_train)
  
  linear <- predict(preg_wf, new_data=test)
  
  exp_lin_preds_2 <- exp(linear)
  
  
  workflow_kaggle_submission <- exp_lin_preds_2 %>%
    bind_cols(., test) %>% 
    select(datetime, .pred) %>% 
    rename(count=.pred) %>% 
    mutate(datetime=as.character(format(datetime))) 
  
  vroom_write(x=workflow_kaggle_submission, file="PenalizedRegression.csv", delim = ",")
  
install.packages("glmnet")
library(glmnet)


#Tuning Parameters
preg_model <- linear_reg(penalty=tune(),
                        mixture=tune()) %>% 
  set_engine("glmnet")
preg_wff <- workflow() %>% 
  add_recipe(my_recipe_2) %>% 
  add_model(preg_model)

grid_of_tuning_parameters <- grid_regular(penalty(),
                                          mixture(),
                                          levels = 7)
folds <- vfold_cv(new_train, v = 5, repeats = 1)

CV_results <- preg_wff %>% 
  tune_grid(resamples = folds,
            grid = grid_of_tuning_parameters,
            metrics = metric_set(rmse, mae, rsq))
collect_metrics(CV_results) %>% 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune <- CV_results %>% 
  select_best(metric = "rmse")

final_wf <-
preg_wff %>%
finalize_workflow(bestTune) %>%
fit(data=new_train)

predict <- final_wf %>%
predict(new_data = test)

exp_lin_preds_3 <- exp(predict)


workflow_kaggle_submission_2 <- exp_lin_preds_3 %>%
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

vroom_write(x=workflow_kaggle_submission_2, file="TuningRegression.csv", delim = ",")


#Regression Trees
install.packages("rpart")
library(tidymodels)

my_model <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

my_recipe_2 <- recipe(count ~ ., data = new_train) %>% 
  step_mutate(weather =ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3)) %>% 
  step_time(datetime, features = c("hour")) %>% 
  step_mutate(datetime_hour = as.factor(datetime_hour))%>%
  step_mutate(season = factor(season, levels = 1:4, labels = c("spring", "summer", "fall" , "winter"))) %>% 
  step_rm(datetime, temp, holiday, workingday) %>% 
  step_mutate(windspeed = ifelse(windspeed == 0, 0, 1))%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

tree_wf <- workflow() %>% 
  add_recipe(my_recipe_2) %>% 
  add_model(my_model)

grid_of_tuning_parameters_2 <- grid_regular(tree_depth(),
                                          cost_complexity(),
                                          min_n(),
                                          levels = 6)
folds <- vfold_cv(new_train, v = 6, repeats = 1)

CV_results_2 <- tree_wf %>% 
  tune_grid(resamples = folds,
            grid = grid_of_tuning_parameters_2,
            metrics = metric_set(rmse, mae, rsq))
collect_metrics(CV_results_2) %>% 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune <- CV_results_2 %>% 
  select_best(metric = "rmse")

final_wf_2 <-
  tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=new_train)

predict <- final_wf_2 %>%
  predict(new_data = test)

exp_lin_preds_3 <- exp(predict)


workflow_kaggle_submission_2 <- exp_lin_preds_3 %>%
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

vroom_write(x=workflow_kaggle_submission_2, file="RegressionTree.csv", delim = ",")



#Random Forrest

rf_model <- rand_forest(mtry = tune(),
                          min_n = tune(),
                        trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

my_recipe_rf <- recipe(count ~ ., data = new_train) %>% 
  step_mutate(weather =ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3)) %>% 
  step_time(datetime, features = c("hour")) %>% 
  step_mutate(datetime_hour = as.factor(datetime_hour))%>%
  step_mutate(season = factor(season, levels = 1:4, labels = c("spring", "summer", "fall" , "winter"))) %>% 
  step_rm(datetime, temp, holiday, workingday) %>% 
  step_mutate(windspeed = ifelse(windspeed == 0, 0, 1))%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

rf_wf <- workflow() %>% 
  add_recipe(my_recipe_rf) %>% 
  add_model(rf_model)

grid_of_tuning_parameters_rf <- grid_regular(mtry(range = c(1,10)),
                                            min_n(),
                                            levels = 3)
folds_rf <- vfold_cv(new_train, v = 3, repeats = 1)

CV_results_rf <- rf_wf %>% 
  tune_grid(resamples = folds_rf,
            grid = grid_of_tuning_parameters_rf,
            metrics = metric_set(rmse, mae, rsq))
collect_metrics(CV_results_rf) %>% 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune_rf <- CV_results_rf %>% 
  select_best(metric = "rmse")

final_wf_rf <-
  rf_wf %>%
  finalize_workflow(bestTune_rf) %>%
  fit(data=new_train)

predict_rf <- final_wf_rf %>%
  predict(new_data = test)

exp_lin_preds_rf <- exp(predict_rf)


workflow_kaggle_submission_rf <- exp_lin_preds_rf %>%
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

vroom_write(x=workflow_kaggle_submission_rf, file="RandomForrest.csv", delim = ",")

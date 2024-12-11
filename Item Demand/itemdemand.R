library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(themis)

train <- vroom("./train.csv")
test <- vroom("./test.csv")
view(train)

train <- filter(train, store == 4) 
train <- filter(train, item == 35)         

my_recipe <- recipe(sales ~ ., data = train) %>%
  step_date(date, features="dow") %>% 
  step_date(date, features="month") %>% 
  step_date(date, features = "year") %>% 
  step_date(date, features = "decimal") %>% 
  step_mutate(date_decimal = as.numeric(date_decimal)) 


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)          

view(baked)

rf_model <- rand_forest(mtry = 10,
                        min_n = 40,
                        trees = 5) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model) 

#grid_of_tuning_parameters_rf <- grid_regular(mtry(range = c(1,10)),
 #                                           min_n(),
  #                                         levels = 2)
#folds_rf <- vfold_cv(train, v = 2, repeats = 2)

#CV_results_rf <- rf_wf %>% 
 #tune_grid(resamples = folds_rf,
  #        grid = grid_of_tuning_parameters_rf,
   #      metrics = metric_set(smape),
    #     control = control_grid(verbose = TRUE))


bestTune_rf <-rf_wf %>% 
 show_best(n = 1, metric = "smape")

print(bestTune_rf)
print(CV_results_rf)

final_wf_rf <-
 rf_wf %>%
finalize_workflow(bestTune_rf) %>%
fit(data=train)

predict_rf <- predict(rf_wf, new_data = test, type = "prob")
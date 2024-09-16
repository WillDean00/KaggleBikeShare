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

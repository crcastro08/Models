library(tidymodels)
library(tidyverse)
library(janitor)
library(doSNOW)

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

#Shutdown cluster
stopCluster(cl)



file_path <- "C:\\Users\\carlos.castro\\OneDrive\\DSI\\DMBA-R-datasets\\LaptopSales.csv"

data_tbl <- read_csv(file_path) %>% clean_names() %>% 
  select(-c("date","configuration","customer_x","customer_y",
            "store_x","store_y","customer_postcode","store_postcode"))

glimpse(data_tbl)
trees_split <- initial_split(data_tbl,prop = 0.5)
trees_train <- training(trees_split)
trees_test <- testing(trees_split)



tree_rec <- recipe(retail_price ~ ., data = trees_train) %>%
step_naomit(all_outcomes(), all_predictors()) %>%
#update_role(tree_id, new_role = "ID") %>%
#step_other(species, caretaker, threshold = 0.01) %>%
#step_other(site_info, threshold = 0.005) %>%
step_dummy(all_nominal(), -all_outcomes()) #%>%
#step_date(date, features = c("year")) %>%
# step_rm(date) %>%
#step_downsample(legal_status)


tree_prep <- prep(tree_rec)
juiced <- juice(tree_prep)


tune_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("ranger")


tune_wf <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(tune_spec)

set.seed(234)
trees_folds <- vfold_cv(trees_train,v =5)

cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = 5
)

stopCluster(cl)
tune_res
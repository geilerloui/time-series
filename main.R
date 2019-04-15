# Packages for business forecasting
# https://www.lancaster.ac.uk/lums/news/r-packages-for-business-forecasting
# to do list:
# 1. Add temps de calcul par algorithme
# 2. Voir ce qu'on peut faire pour les events spéciaux (la cup de golf et l'autre blocs noel et cie)
# 3. Peut être mettre une option sur la période sur laquel on veut travailler
# 4. Ajout des calculs de scoring (done)
# 5. Forcément delve deeper in the algorithms
# 6. checker les graphes de décomposition (autre que prophet) et les mettre en png qq part ?
# 7. et le truc de détection du type de demand
# ---------------
# rajouter un paramètre fréquence pour laisser le choix de regrouper les data par jours ou par semaine

library(dplyr)
library(tidyr)

# -------------------------------------------------------------------------------------
# ------------------- Preprocesing ----------------------------------------------------
# -------------------------------------------------------------------------------------
# read file
file = "slots.csv"
file = "slots_032019.csv"
df <- read.csv(file)

# Select some columns
cols = c("job", "starts_at", "minute_duration", "status", "fr_full_id", "category")
df <- df %>% select(cols)

# Remove all canceled mission then we can remove the status column
df <- df %>% filter(status != "canceled")
df[, c("status")] <- list(NULL)

# Split fr_full_id in three and remove x1 and x2 which are not interesting
df <- df %>% separate("fr_full_id", c("x1", "category", "x2"), sep=":")
df[, c("x1", "x2")] <- list(NULL)

# Split the starts_at column into
df <- df %>% separate("starts_at", c("timestamp", "time_hours"), sep=" ")
df[, c("time_hours")] <- list(NULL)

# groupby timestamp and sum minute_duration and convert back to DF othw it's of tibble type
df <- df %>% group_by(timestamp, job, category) %>% summarize(sum(minute_duration))
df <- as.data.frame(df)

# Rename column 
df <- df %>% rename("minute_duration" = "sum(minute_duration)")

# Changee type of date
df$timestamp <- as.Date(df$timestamp, "%Y-%m-%d")

# -------------------------------------------------------------------------------------
# ------------------- Statistical analysis --------------------------------------------
# -------------------------------------------------------------------------------------
library(ggplot2)

# Let's focus on the waiter job
df_waiter <- df %>% filter(job == "runner")
df_waiter[, c("job", "category")] <- list(NULL)

# Descr stat - with theme (white background)
theme_set(theme_bw())
ggplot(df_waiter, aes(x = timestamp)) + geom_line(aes(y = minute_duration, col="waiter"))

# -------------------------------------------------------------------------------------
# ------------------- Change timestamp --------------------------------------------
# -------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------
# ------------------- Modelisation ----------------------------------------------------
# -------------------------------------------------------------------------------------

library(forecast)
library(prophet)
library(forecastxgb)
library(nnfor)
library(MAPA)
library(tsintermittent)

# Split data into training and test
training_size <- 0.8

train_size <- nrow(df_waiter) * training_size
train <- df_waiter[1: train_size, ]
test <- df_waiter[train_size:nrow(df_waiter), ]



# Scoring ------------------------------------------------------------

library(Metrics)

calculate_scoring <- function(y, yhat) {
  MAE <- mae(actual = y, predicted = yhat)
  MAPE <- mape(actual = y, predicted = yhat)
  RMSE <- rmse(actual = y, predicted = yhat)
  return(rbind(MAE, MAPE, RMSE))
}

# We define the scores dataframe: it will stores all the values
scores <- data.frame()


# 2. ARIMA ------------------------------------------------------------
# how to not get a flat line for arima
# https://stats.stackexchange.com/questions/286900/arima-forecast-straight-line
# Modelisation with ARIMA

train <- train %>% rename("ds" = "timestamp", "y" = "minute_duration")
test <- test %>% rename("ds" = "timestamp", "y" = "minute_duration")

arima_train <- train
arima_train[, c("ds")] <- NULL
arima_test <- test
arima_test[, c("ds")] <- NULL

# Convert DF to timeseries (ts); D=1, we force seasonality
arima_train_ts <- ts(arima_train,frequency=365)

# tsoutlier ----------------------------------------------------------
library(tsoutliers)

outliers_arima_train_ts <- tso(arima_train_ts)

plot(outliers_arima_train_ts)

arima_train_ts <- outliers_arima_train_ts$yadj

train$y <- arima_train_ts

# go back to arima --------------------------------------------------

fit_arima <- auto.arima(arima_train_ts, D=1)
fcast_arima <- forecast(fit_arima, h=nrow(arima_test))

# If you want to get the parameters
arimaorder(fit_arima)

plot(fcast_arima)

s1 <- calculate_scoring(fcast_arima$mean, arima_test$y)
colnames(s1) <- "arima"

# 1. PROPHET ------------------------------------------------------------

fit_prophet <- prophet(train)
future <- make_future_dataframe(fit_prophet, periods = nrow(test))
fcast_prophet <- predict(fit_prophet, future)

plot(fit_prophet, fcast_prophet)
prophet_plot_components(fit_prophet, fcast_prophet)


s <- calculate_scoring(test$y, tail(fcast_prophet$yhat, nrow(test)))
colnames(s) <- "prophet"

# 3. XGBOOST ------------------------------------------------------------

# No preprocessing, just mandatory that the data is in ts format
fit_xgboost <- xgbar(arima_train_ts)
fcast_xgboost <- forecast(fit_xgboost, h=nrow(arima_test))

plot(fcast_xgboost)

s2 <- calculate_scoring(fcast_xgboost$mean, arima_test$y)
colnames(s2) <- "xgboost"

# 4. NNETR (Feedforward NN) ------------------------------------------

fit_nnetr <- nnetar(arima_train_ts)
fcast_nnetr <- forecast(fit_nnetr, h=nrow(arima_test))

# add accuracy
s3 <- calculate_scoring(fcast_nnetr$mean, arima_test$y)
colnames(s3) <- "nnetr"

plot(fcast_nnetr)

# # 5. MLP ------------------------------------------
# fit_mlp <- mlp(arima_train_ts)
# fcast_mlp <- forecast(fit_mlp, h=nrow(arima_test))
# 
# # add accuracy
# s4 <- calculate_scoring(fcast_mlp$mean, arima_test$y)
# colnames(s4) <- "mlp"
# 
# plot(fcast_mlp$mean)
# 
# # plot neural network
# plot(fit_mlp)

# 5. ETS ------------------------------------------

fit_ets <- tbats(arima_train_ts)
fcast_ets <- forecast(fit_ets, h=nrow(arima_test))

# add accuracy
s5 <- calculate_scoring(fcast_ets$mean, arima_test$y)
colnames(s5) <- "ets"

plot(fcast_ets$mean)


###### Scores

# result_all_scores <- cbind(s, s1, s2, s3, s4, s5)
result_all_scores <- cbind(s, s1, s2, s3, s5)

# -------------------------------------------------------------------------------------
# ------------------- Model competition --------------------------------------------

# Merge all time series
df_all <- cbind(test, 
      as.data.frame(fcast_xgboost$mean) %>% rename("y_hat_xgboost" = "x"), 
      as.data.frame(fcast_nnetr$mean) %>% rename("y_hat_nnetar" = "x"),
      as.data.frame(tail(fcast_prophet$yhat, nrow(test))) %>% rename("y_hat_prophet" = "tail(fcast_prophet$yhat, nrow(test))"),
      as.data.frame(fcast_arima$mean) %>% rename("y_hat_arima" = "x"),
      as.data.frame(fcast_ets$mean) %>% rename("y_hat_ets" = "x")
      )

# Convert the DF to another format for plotting 
# With the tidyr package "gather"
nb_of_algorithm <- 6
gather_df <- df_all %>% gather(key="Algorithm", "values", 2:7)

unique(gather_df$Algorithm)

# Solution: https://stackoverflow.com/questions/55610018/multiple-time-series-with-face-wrap/55610223#55610223

ggplot(gather_df %>% filter(Algorithm != "y"), aes(x = ds, y = values)) +
  geom_line(aes(color = Algorithm)) +
  scale_color_brewer(palette = "Dark2") +
  facet_wrap(~ Algorithm) + 
  geom_line(data = gather_df %>% filter(Algorithm == "y") %>% select(-Algorithm))


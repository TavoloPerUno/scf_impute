library(missForest)
library(doParallel)

setwd("~/Documents/pyWorkspace/imputation/r")

df_raw <- read.csv('../data/raw_2013.csv')

df_col_structure <- read.csv('../data/col_structure.csv', stringsAsFactors=FALSE)

df_raw[unlist(strsplit(df_col_structure$year_col[1], ","))][is.na(df_raw[unlist(strsplit(df_col_structure$year_col[1], ","))])] <- -1

df_raw <- df_raw[c(unlist(strsplit(df_col_structure$year_col[1], ",")),
                   unlist(strsplit(df_col_structure$char_col[1], ",")),
                   unlist(strsplit(df_col_structure$num_col[1], ",")))]

df_char_cols <- df_raw[,unlist(strsplit(df_col_structure$char_col[1], ","))]

df_char_cols[] <- lapply(df_char_cols, as.factor)

df_num_cols <- df_raw[,c(unlist(strsplit(df_col_structure$num_col[1], ",")),
                         unlist(strsplit(df_col_structure$year_col[1], ",")))]

df_num_cols[] <- lapply(df_num_cols, as.numeric)

df_raw[colnames(df_char_cols)] <- df_char_cols
df_raw[colnames(df_num_cols)] <- df_num_cols


registerDoParallel(cores = 60)
imputed <- NA
imputed <- missForest(xmis = df_raw,
                      ntree = 100,
                      parallelize = 'forests')


save.image(file='../data/myEnvironment.RData')

write.csv(imputed, file='../data/missforest_imputed.csv')
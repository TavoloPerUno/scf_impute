library(missForest)
library(doParallel)
library(dplyr)
require(data.table)

#setwd("~/Documents/pyWorkspace/imputation/r")

df_withhold <- read.csv('../data/withheld_1.csv')

df_withhold <- df_withhold %>%
  mutate_all(as.character)

load(file='../data/myEnvironment.rdata')

df_imputed <- imputed$ximp

withhold <- function(column){ 
  x <- df_imputed[, column]
  x[as.numeric(unlist(strsplit(df_withhold[,column][1], ",")))] <- NA
  return (x)
}

df_imputed[,colnames(df_withhold[,!is.na(df_withhold)])] <- lapply(colnames(df_withhold[,!is.na(df_withhold)]), withhold)

df_col_structure <- read.csv('../data/col_structure.csv', stringsAsFactors=FALSE)

# df_raw[unlist(strsplit(df_col_structure$year_col[1], ","))][is.na(df_raw[unlist(strsplit(df_col_structure$year_col[1], ","))])] <- -1
# 
# df_raw <- df_raw[c(unlist(strsplit(df_col_structure$year_col[1], ",")),
#                    unlist(strsplit(df_col_structure$char_col[1], ",")),
#                    unlist(strsplit(df_col_structure$num_col[1], ",")))]
# 
df_raw <- df_imputed
char_cols <- intersect(unlist(strsplit(df_col_structure$char_col[1], ",")), colnames(df_raw))
num_cols <- intersect(c(unlist(strsplit(df_col_structure$num_col[1], ",")),
                        unlist(strsplit(df_col_structure$year_col[1], ","))), colnames(df_raw))

df_char_cols <- df_raw[,char_cols]

df_char_cols[] <- lapply(df_char_cols, as.factor)

df_num_cols <- df_raw[,num_cols]

df_num_cols[] <- lapply(df_num_cols, as.numeric)

df_raw[colnames(df_char_cols)] <- df_char_cols
df_raw[colnames(df_num_cols)] <- df_num_cols

registerDoParallel(cores = 24)
imputed <- NA
imputed <- missForest(xmis = df_raw,
                      ntree = 100,
                      parallelize = 'forests')


save.image(file='../data/iter_1.RData')

write.csv(imputed$ximp, file='../data/missforest_imputed_1.csv')
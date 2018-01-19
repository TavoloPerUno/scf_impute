library(missForest)
library(doParallel)

#setwd("~/Documents/pyWorkspace/imputation/r")

df_raw <- read.csv('../data/withheld_cleaned.csv')

df_col_structure <- read.csv('../data/col_structure.csv', stringsAsFactors=FALSE)

# df_raw[unlist(strsplit(df_col_structure$year_col[1], ","))][is.na(df_raw[unlist(strsplit(df_col_structure$year_col[1], ","))])] <- -1
# 
# df_raw <- df_raw[c(unlist(strsplit(df_col_structure$year_col[1], ",")),
#                    unlist(strsplit(df_col_structure$char_col[1], ",")),
#                    unlist(strsplit(df_col_structure$num_col[1], ",")))]
# 

char_cols <- intersect(unlist(strsplit(df_col_structure$char_col[1], ",")), colnames(df_raw))
num_cols <- intersect(c(unlist(strsplit(df_col_structure$num_col[1], ",")),
                        unlist(strsplit(df_col_structure$year_col[1], ","))), colnames(df_raw))

df_char_cols <- df_raw[,char_cols]

df_char_cols[] <- lapply(df_char_cols, as.factor)

df_num_cols <- df_raw[,num_cols]

df_num_cols[] <- lapply(df_num_cols, as.numeric)

df_raw[colnames(df_char_cols)] <- df_char_cols
df_raw[colnames(df_num_cols)] <- df_num_cols


registerDoParallel(cores = 16)
imputed <- NA
imputed <- missForest(xmis = df_raw,
                      ntree = 100,
                      parallelize = 'forests')


save.image(file='../data/myEnvironment.RData')

write.csv(imputed, file='../data/missforest_imputed.csv')
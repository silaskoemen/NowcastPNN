#install.packages("read.dbc") #You need this package
library("read.dbc")
setwd("/Users/silaskoemen/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Imperial/Thesis/src")


dbc2dbf <- function(rawDir, convertedDir, file) {
  #file <- paste0(file, ".dbc")
  # reads dbc file
  x <- read.dbc(paste(rawDir, paste0(file, ".dbc"), sep=""))
  # write it to csv
  write.csv(x, file=paste(convertedDir, file, ".csv", sep=""))
}

args = commandArgs(trailingOnly=TRUE)
try(dbc2dbf(args[1], args[2], args[3]), TRUE)


#dfs <- read.dbc("../data/raw/Dengue/DENGAC13.dbc")

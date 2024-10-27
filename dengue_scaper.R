library(tidyverse)
library(lubridate)
library(nowcaster)
library(microdatasus)

## To see Nowcasting as if we were on the verge of rise in the curve
data("sragBH")
srag_now<-sragBH |> 
  filter(DT_DIGITA <= "2020-07-04")

dengue <- fetch_datasus(2021, year_end = 2024, information_system = "SINAN-DENGUE")

require(tidyverse)
require(data.table)

df_mkt <- fread(input = "./data/preprocessed/mkt_anonymized_data.csv",data.table = F)
df_cpn <- fread(input = "./data/preprocessed/cpn_anonymized_data.csv",data.table = F)
df_mkt$domain <- 'MKT'
df_cpn$domain <- 'CPN'

df_full <- bind_rows(df_mkt, df_cpn) %>% mutate(
  domain = factor(domain, levels = c('MKT','CPN'))
)


#actuarial kpi

actuarial_kpi_ds <- group_by(df_full, year, domain) %>% summarise(
  exposures = sum(exposure),
  claims = sum(claims),frequency = claims/exposures
) 

# creazione delle descrittive

require(gtsummary)


vars_cont <- grep(pattern = "cont",x = colnames(df_full),value = T)
vars_cat <- setdiff(grep(pattern = "cat",x = colnames(df_full),value = T), "cat1")

table_descr <- dplyr::select(df_full, all_of(c(vars_cont,vars_cat,'domain'))) %>%    tbl_summary(
  by=domain,
  type = list(cat2 ~ "categorical", cat6 ~ "categorical",cat7~ "categorical", cat8~ "categorical"),
  statistic = list(all_continuous() ~ "{mean} ({sd}) {min} {max}",
                   all_categorical() ~ "{n} / {N} ({p}%)"),
  digits = all_continuous() ~ 3, missing_text = "(Missing)"
) %>% add_n() #%>% add_p() 


save(list=c('table_descr','actuarial_kpi_ds'),file = "./output/descriptives.RData")

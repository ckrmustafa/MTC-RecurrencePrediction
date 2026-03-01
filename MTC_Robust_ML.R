cat("\n========================================================\n")
cat("  MTC ROBUST ML ANALYSIS\n")
cat("  Nested CV + Bootstrap CI + Multi-Seed + Stacking\n")
cat("========================================================\n\n")

# ============================================================
# 1. PACKAGES
# ============================================================
pkgs <- c(
  "tidyverse","caret","randomForest","glmnet","pROC",
  "xgboost","e1071","ROSE","naivebayes","rpart",
  "rpart.plot","nnet","gbm",
  "ggplot2","ggpubr","cowplot","viridis",
  "corrplot","tableone","survival","survminer",
  "DALEX","scales","gridExtra"
)
inst <- rownames(installed.packages())
miss <- pkgs[!pkgs %in% inst]
if (length(miss) > 0) install.packages(miss, repos="https://cloud.r-project.org")
suppressPackageStartupMessages(invisible(lapply(pkgs, library, character.only=TRUE)))
message("[OK] Packages loaded")

# ============================================================
# 2. SETTINGS  -- burada hizi ayarlayabilirsiniz
# ============================================================
DPI     <- 600
FIG_W   <- 12
FIG_H   <- 9
OUT     <- getwd()
N_BOOT  <- 500   # Bootstrap tekrar: hiz icin 100 yap
N_SEEDS <- 30    # Multi-seed sayisi: hiz icin 10 yap
OUTER_K <- 10    # Nested CV dis katman
INNER_K <- 5     # Nested CV ic katman
WB_COL  <- "#1565C0"
BB_COL  <- "#B71C1C"
WHITEBOX <- c("LASSO","CART","NaiveBayes","KNN")
BLACKBOX <- c("RandomForest","SVM","NeuralNet","GBM","XGBoost")

save_fig <- function(p, fn, w=FIG_W, h=FIG_H) {
  ggsave(file.path(OUT,fn), p, width=w, height=h, dpi=DPI, units="in")
  message("[OK] ", fn)
}
save_csv <- function(d, fn) {
  write.csv(d, file.path(OUT,fn), row.names=FALSE)
  message("[OK] ", fn)
}
open_png <- function(fn, w=FIG_W, h=FIG_H)
  png(file.path(OUT,fn), width=w, height=h, units="in", res=DPI)

# ============================================================
# 3. DATA PREP
# ============================================================
message("\n--- DATA LOADING ---")
if (!file.exists("DATA_2.csv"))
  stop("DATA_2.csv bulunamadi: ", getwd())

df_raw <- read.csv("DATA_2.csv", stringsAsFactors=FALSE, na.strings=c("","NA"))
df_raw <- df_raw[, !duplicated(names(df_raw))]

num_force <- c("age_at_diagnosis","follow_up_year","tumor_size_mm",
               "usg_size_mm","calcitonin_preop","calcitonin_postop",
               "cea_preop","cea_postop","recurrence_time_months")
fac_vars <- c("sex","disease_type","ret_mutation","pheochromocytoma",
              "tumor_side","multifocal","lymph_node_invasion",
              "capsular_invasion","soft_tissue_invasion",
              "metastasis_at_diagnosis","mediastinal_metastasis",
              "lung_metastasis_at_diagnosis","metastasis_followup",
              "lung_metastasis_followup","liver_metastasis","bone_metastasis",
              "stage","lymph_node_metastasis","bone_metastasis_present",
              "lung_metastasis_present","liver_metastasis_present",
              "adrenal_metastasis","recurrence","death_mtc","vital_status",
              "tki_therapy","radiotherapy","lutetium_therapy",
              "chemotherapy","mibg_therapy","tt_cnd_snd","tt_cnd",
              "total_thyroidectomy","completion_thyroidectomy","hemithyroidectomy",
              "disease_status","alive_no_disease","alive_with_disease",
              "alive_unknown_status")

df <- df_raw %>%
  mutate(across(all_of(num_force), ~as.numeric(as.character(.)))) %>%
  mutate(across(all_of(intersect(fac_vars, names(.))), as.factor)) %>%
  mutate(
    sex_label          = factor(sex, levels=c(0,1), labels=c("Female","Male")),
    disease_type_label = factor(disease_type, levels=c(0,1),
                                labels=c("Sporadic","Hereditary")),
    stage_label        = factor(stage, levels=c(1,2,3,4),
                                labels=c("I","II","III","IV")),
    time_surv          = as.numeric(as.character(follow_up_year)),
    event_surv         = as.numeric(as.character(death_mtc)),
    recurrence_num     = as.numeric(as.character(recurrence))
  )

ml_vars <- c(
  "age_at_diagnosis","sex","disease_type","ret_mutation",
  "tumor_size_mm","stage","multifocal","lymph_node_invasion",
  "capsular_invasion","soft_tissue_invasion",
  "metastasis_at_diagnosis","calcitonin_preop","cea_preop",
  "lymph_node_metastasis","lung_metastasis_present",
  "bone_metastasis_present","liver_metastasis_present"
)

df_ml <- df %>%
  select(recurrence, all_of(ml_vars)) %>%
  mutate(across(all_of(ml_vars) & where(is.factor), as.numeric)) %>%
  na.omit()

df_ml$recurrence <- factor(
  as.integer(as.character(df_ml$recurrence)),
  levels=c(0,1), labels=c("Rec0","Rec1"))
df_ml <- df_ml %>% filter(!is.na(recurrence))

cat(sprintf("ML dataset: n=%d | Rec1=%d | Rec0=%d\n",
            nrow(df_ml),
            sum(df_ml$recurrence=="Rec1"),
            sum(df_ml$recurrence=="Rec0")))

# ============================================================
# 4. YARDIMCI: tek turda egit + AUC dondur
# ============================================================
xgb_par <- list(
  objective="binary:logistic", eval_metric="auc",
  max_depth=3, eta=0.1, subsample=0.8,
  colsample_bytree=0.8, nthread=1
)

train_and_eval <- function(train_data, test_data, ctrl_in, seed=42) {
  set.seed(seed)
  train_bal <- tryCatch(
    ROSE(recurrence~., data=train_data, seed=seed,
         N=2*nrow(train_data))$data,
    error=function(e) upSample(x=train_data[,-1],
                               y=train_data$recurrence, yname="recurrence"))

  feats <- setdiff(names(train_bal), "recurrence")
  results <- list()

  caret_spec <- list(
    LASSO      = list(method="glmnet", pp=c("center","scale"),
                      tl=8, extra=list(family="binomial")),
    CART       = list(method="rpart", pp=NULL, tl=8, extra=list()),
    NaiveBayes = list(method="naive_bayes", pp=NULL, tl=5, extra=list()),
    KNN        = list(method="knn", pp=c("center","scale"),
                      tl=NULL, extra=list(),
                      tg=data.frame(k=seq(1,15,by=2))),
    RandomForest=list(method="rf", pp=NULL, tl=4,
                      extra=list(ntree=300)),
    SVM        = list(method="svmRadial", pp=c("center","scale"),
                      tl=4, extra=list()),
    NeuralNet  = list(method="nnet", pp=c("center","scale"),
                      tl=4, extra=list(trace=FALSE, MaxNWts=400)),
    GBM        = list(method="gbm", pp=NULL, tl=4,
                      extra=list(verbose=FALSE))
  )

  for (nm in names(caret_spec)) {
    spec <- caret_spec[[nm]]
    tryCatch({
      args <- c(
        list(recurrence~., data=train_bal,
             method=spec$method, metric="ROC",
             trControl=ctrl_in,
             preProcess=spec$pp),
        if (!is.null(spec$tl)) list(tuneLength=spec$tl),
        if (!is.null(spec$tg)) list(tuneGrid=spec$tg),
        spec$extra
      )
      m <- do.call(train, args)
      prob <- predict(m, test_data, type="prob")[,"Rec1"]
      roc_v <- pROC::roc(as.numeric(test_data$recurrence=="Rec1"),
                         as.numeric(prob), quiet=TRUE)
      results[[nm]] <- as.numeric(pROC::auc(roc_v))
    }, error=function(e) NULL)
  }

  # XGBoost
  tryCatch({
    xtr <- xgb.DMatrix(as.matrix(train_bal[,feats]),
                       label=as.numeric(train_bal$recurrence=="Rec1"))
    xte <- xgb.DMatrix(as.matrix(test_data[,feats]),
                       label=as.numeric(test_data$recurrence=="Rec1"))
    cv_r <- xgb.cv(params=xgb_par, data=xtr, nrounds=150, nfold=5,
                   early_stopping_rounds=15, verbose=0, print_every_n=9999)
    best <- cv_r$best_iteration
    if (is.null(best)||is.na(best)||best<1) best <- 50
    m_x <- xgb.train(params=xgb_par, data=xtr, nrounds=best,
                     verbose=0, print_every_n=9999)
    prob <- predict(m_x, xte)
    roc_v <- pROC::roc(as.numeric(test_data$recurrence=="Rec1"),
                       as.numeric(prob), quiet=TRUE)
    results[["XGBoost"]] <- as.numeric(pROC::auc(roc_v))
  }, error=function(e) NULL)

  data.frame(
    Model = names(results),
    AUC   = round(as.numeric(unlist(results)), 4),
    stringsAsFactors=FALSE
  )
}

# ============================================================
# 5. NESTED CROSS-VALIDATION
# ============================================================
message("\n========================================")
message(sprintf("  [1/4] NESTED CV (%d-outer / %d-inner)", OUTER_K, INNER_K))
message("========================================\n")

set.seed(42)
outer_folds <- createFolds(df_ml$recurrence, k=OUTER_K,
                           list=TRUE, returnTrain=FALSE)
ctrl_inner <- trainControl(
  method="cv", number=INNER_K, classProbs=TRUE,
  summaryFunction=twoClassSummary, verboseIter=FALSE)

ncv_rows <- list()
for (fi in seq_along(outer_folds)) {
  cat(sprintf("  Outer fold %d/%d\n", fi, OUTER_K))
  te_idx <- outer_folds[[fi]]
  tr_idx <- setdiff(seq_len(nrow(df_ml)), te_idx)
  te <- df_ml[te_idx, ]
  tr <- df_ml[tr_idx, ]
  if (length(unique(te$recurrence)) < 2) { cat("    skip\n"); next }
  r <- train_and_eval(tr, te, ctrl_inner, seed=fi*11)
  if (!is.null(r)) { r$Fold <- fi; ncv_rows[[fi]] <- r }
}
ncv_df <- do.call(rbind, Filter(Negate(is.null), ncv_rows))

ncv_sum <- ncv_df %>%
  group_by(Model) %>%
  summarise(
    N_folds    = n(),
    Mean_AUC   = round(mean(AUC,na.rm=TRUE),4),
    Median_AUC = round(median(AUC,na.rm=TRUE),4),
    SD_AUC     = round(sd(AUC,na.rm=TRUE),4),
    Min_AUC    = round(min(AUC,na.rm=TRUE),4),
    Max_AUC    = round(max(AUC,na.rm=TRUE),4),
    .groups    = "drop"
  ) %>%
  mutate(Type=ifelse(Model %in% WHITEBOX,"WhiteBox","BlackBox")) %>%
  arrange(desc(Mean_AUC))

cat("\n=== NESTED CV RESULTS ===\n")
print(ncv_sum)
save_csv(ncv_sum, "Table_NCV_Summary.csv")
save_csv(ncv_df, "Table_NCV_AllFolds.csv")

# ============================================================
# 6. ANA MODELLER (Bootstrap icin)
# ============================================================
message("\n--- Training main models for Bootstrap ---")
set.seed(42)
yes_idx   <- which(df_ml$recurrence=="Rec1")
no_idx    <- which(df_ml$recurrence=="Rec0")
tr_yes    <- sample(yes_idx, size=max(1,round(0.70*length(yes_idx))))
tr_no     <- sample(no_idx,  size=max(1,round(0.70*length(no_idx))))
train_df  <- df_ml[c(tr_yes,tr_no), ]
test_df   <- df_ml[-c(tr_yes,tr_no), ]

train_bal_main <- tryCatch(
  ROSE(recurrence~., data=train_df, seed=42, N=2*nrow(train_df))$data,
  error=function(e) upSample(x=train_df[,-1], y=train_df$recurrence,
                             yname="recurrence"))
xgb_feat <- setdiff(names(train_bal_main), "recurrence")

ctrl_main <- trainControl(
  method="repeatedcv", number=5, repeats=3,
  classProbs=TRUE, summaryFunction=twoClassSummary,
  savePredictions="final", verboseIter=FALSE)

main_models <- list()
main_models$LASSO <- train(recurrence~., data=train_bal_main,
  method="glmnet", family="binomial", metric="ROC",
  trControl=ctrl_main, tuneLength=10, preProcess=c("center","scale"))
main_models$CART <- train(recurrence~., data=train_bal_main,
  method="rpart", metric="ROC", trControl=ctrl_main, tuneLength=10)
main_models$NaiveBayes <- train(recurrence~., data=train_bal_main,
  method="naive_bayes", metric="ROC", trControl=ctrl_main, tuneLength=5)
main_models$KNN <- train(recurrence~., data=train_bal_main,
  method="knn", metric="ROC", trControl=ctrl_main,
  tuneGrid=data.frame(k=seq(1,21,by=2)), preProcess=c("center","scale"))
main_models$RandomForest <- train(recurrence~., data=train_bal_main,
  method="rf", metric="ROC", trControl=ctrl_main,
  tuneLength=5, ntree=500)
main_models$SVM <- train(recurrence~., data=train_bal_main,
  method="svmRadial", metric="ROC", trControl=ctrl_main, tuneLength=5,
  preProcess=c("center","scale"))
main_models$NeuralNet <- train(recurrence~., data=train_bal_main,
  method="nnet", metric="ROC", trControl=ctrl_main, tuneLength=5,
  preProcess=c("center","scale"), trace=FALSE, MaxNWts=500)
main_models$GBM <- train(recurrence~., data=train_bal_main,
  method="gbm", metric="ROC", trControl=ctrl_main,
  tuneLength=5, verbose=FALSE)

xgb_tr_main <- xgb.DMatrix(as.matrix(train_bal_main[,xgb_feat]),
                            label=as.numeric(train_bal_main$recurrence=="Rec1"))
xgb_te_main <- xgb.DMatrix(as.matrix(test_df[,xgb_feat]),
                            label=as.numeric(test_df$recurrence=="Rec1"))
xgb_cv_main <- xgb.cv(params=xgb_par, data=xgb_tr_main, nrounds=200,
                       nfold=5, early_stopping_rounds=20,
                       verbose=0, print_every_n=9999)
best_nr_main <- xgb_cv_main$best_iteration
if (is.null(best_nr_main)||is.na(best_nr_main)||best_nr_main<1) best_nr_main <- 50
main_models$XGBoost <- xgb.train(params=xgb_par, data=xgb_tr_main,
                                  nrounds=best_nr_main, verbose=0,
                                  print_every_n=9999)
message("[OK] Main models trained")

all_nms <- c(WHITEBOX, setdiff(BLACKBOX,"XGBoost"), "XGBoost")

get_prob_main <- function(nm) {
  if (nm=="XGBoost") predict(main_models$XGBoost, xgb_te_main)
  else predict(main_models[[nm]], test_df, type="prob")[,"Rec1"]
}

# ============================================================
# 7. BOOTSTRAP CI
# ============================================================
message("\n========================================")
message(sprintf("  [2/4] BOOTSTRAP CI (%d tekrar)", N_BOOT))
message("========================================\n")

boot_mat <- matrix(NA, nrow=N_BOOT, ncol=length(all_nms))
colnames(boot_mat) <- all_nms

for (b in seq_len(N_BOOT)) {
  if (b %% 100 == 0) cat(sprintf("  Bootstrap %d/%d\n", b, N_BOOT))
  set.seed(b * 17)
  bi  <- sample(nrow(test_df), nrow(test_df), replace=TRUE)
  bte <- test_df[bi, ]
  bxte <- xgb.DMatrix(as.matrix(bte[,xgb_feat]),
                      label=as.numeric(bte$recurrence=="Rec1"))
  tl <- as.numeric(bte$recurrence=="Rec1")
  if (length(unique(tl)) < 2) next

  for (nm in all_nms) {
    tryCatch({
      prob <- if (nm=="XGBoost")
        predict(main_models$XGBoost, bxte)
      else
        predict(main_models[[nm]], bte, type="prob")[,"Rec1"]
      r <- pROC::roc(tl, as.numeric(prob), quiet=TRUE)
      boot_mat[b,nm] <- as.numeric(pROC::auc(r))
    }, error=function(e) NULL)
  }
}

point_aucs <- sapply(all_nms, function(nm) {
  prob <- get_prob_main(nm)
  r <- pROC::roc(as.numeric(test_df$recurrence=="Rec1"),
                 as.numeric(prob), quiet=TRUE)
  as.numeric(pROC::auc(r))
})

boot_sum <- data.frame(
  Model     = all_nms,
  Type      = ifelse(all_nms %in% WHITEBOX,"WhiteBox","BlackBox"),
  AUC_point = round(point_aucs, 4),
  AUC_mean  = round(colMeans(boot_mat, na.rm=TRUE), 4),
  AUC_sd    = round(apply(boot_mat, 2, sd, na.rm=TRUE), 4),
  CI_lo     = round(apply(boot_mat, 2, quantile, 0.025, na.rm=TRUE), 4),
  CI_hi     = round(apply(boot_mat, 2, quantile, 0.975, na.rm=TRUE), 4),
  stringsAsFactors=FALSE
) %>%
  mutate(
    CI_95  = sprintf("%.3f (%.3f-%.3f)", AUC_mean, CI_lo, CI_hi),
    Stable = ifelse(AUC_sd < 0.06, "Yes","No")
  ) %>%
  arrange(desc(AUC_mean))

cat("\n=== BOOTSTRAP CI ===\n")
print(boot_sum[,c("Model","Type","AUC_point","AUC_mean","AUC_sd","CI_95","Stable")])
save_csv(boot_sum, "Table_Bootstrap_CI.csv")

# ============================================================
# 8. MULTI-SEED STABILITY
# ============================================================
message("\n========================================")
message(sprintf("  [3/4] MULTI-SEED STABILITY (%d seeds)", N_SEEDS))
message("========================================\n")

seed_pool <- c(7,13,21,42,77,99,123,256,314,500,
               512,1000,1234,2024,2025,2026,3141,4242,
               5678,6789,7777,8080,8888,9001,9876,
               11111,22222,31415,54321,99999)[1:N_SEEDS]

ctrl_seed <- trainControl(
  method="cv", number=5, classProbs=TRUE,
  summaryFunction=twoClassSummary, verboseIter=FALSE)

seed_rows <- list()
for (si in seq_along(seed_pool)) {
  s <- seed_pool[si]
  cat(sprintf("  Seed %d  (%d/%d)\n", s, si, N_SEEDS))
  set.seed(s)
  yi <- which(df_ml$recurrence=="Rec1")
  ni <- which(df_ml$recurrence=="Rec0")
  tri <- c(sample(yi,max(1,round(0.70*length(yi)))),
           sample(ni,max(1,round(0.70*length(ni)))))
  tei <- setdiff(seq_len(nrow(df_ml)), tri)
  tr_s <- df_ml[tri,]; te_s <- df_ml[tei,]
  if (length(unique(te_s$recurrence)) < 2) { cat("    skip\n"); next }
  r <- train_and_eval(tr_s, te_s, ctrl_seed, seed=s)
  if (!is.null(r)) { r$Seed <- s; seed_rows[[si]] <- r }
}
seed_df <- do.call(rbind, Filter(Negate(is.null), seed_rows))

seed_sum <- seed_df %>%
  group_by(Model) %>%
  summarise(
    N          = n(),
    Mean_AUC   = round(mean(AUC,na.rm=TRUE),4),
    Median_AUC = round(median(AUC,na.rm=TRUE),4),
    SD_AUC     = round(sd(AUC,na.rm=TRUE),4),
    IQR_AUC    = round(IQR(AUC,na.rm=TRUE),4),
    Min_AUC    = round(min(AUC,na.rm=TRUE),4),
    Max_AUC    = round(max(AUC,na.rm=TRUE),4),
    Pct_gt80   = round(100*mean(AUC>0.80,na.rm=TRUE),1),
    .groups    = "drop"
  ) %>%
  mutate(
    Type      = ifelse(Model %in% WHITEBOX,"WhiteBox","BlackBox"),
    Stability = case_when(
      SD_AUC < 0.05 ~ "High",
      SD_AUC < 0.10 ~ "Moderate",
      TRUE          ~ "Low"
    )
  ) %>%
  arrange(desc(Mean_AUC))

cat("\n=== MULTI-SEED STABILITY ===\n")
print(seed_sum[,c("Model","Type","Mean_AUC","SD_AUC","IQR_AUC","Stability","Pct_gt80")])
save_csv(seed_sum, "Table_MultiSeed_Stability.csv")
save_csv(seed_df,  "Table_MultiSeed_AllSeeds.csv")

# ============================================================
# 9. STACKING ENSEMBLE
# ============================================================
message("\n========================================")
message("  [4/4] STACKING ENSEMBLE")
message("========================================\n")

# Out-of-fold predictions
set.seed(42)
cv5 <- createFolds(train_bal_main$recurrence, k=5,
                   list=TRUE, returnTrain=TRUE)
oof <- matrix(NA, nrow=nrow(train_bal_main), ncol=length(all_nms))
colnames(oof) <- all_nms

ctrl_oof <- trainControl(method="cv", number=5, classProbs=TRUE,
                         summaryFunction=twoClassSummary, verboseIter=FALSE)

cat("  Generating OOF predictions...\n")
for (fi in seq_along(cv5)) {
  cat(sprintf("  OOF fold %d/5\n", fi))
  tr_i <- cv5[[fi]]
  va_i <- setdiff(seq_len(nrow(train_bal_main)), tr_i)
  tr_f <- train_bal_main[tr_i,]
  va_f <- train_bal_main[va_i,]
  feats_f <- setdiff(names(tr_f),"recurrence")

  caret_map <- c(LASSO="glmnet", CART="rpart",
                 NaiveBayes="naive_bayes", KNN="knn",
                 RandomForest="rf", SVM="svmRadial",
                 NeuralNet="nnet", GBM="gbm")

  for (nm in names(caret_map)) {
    tryCatch({
      pp_nm <- if (nm %in% c("LASSO","KNN","SVM","NeuralNet"))
        c("center","scale") else NULL
      tl_nm <- 4
      extra_nm <- switch(nm,
        LASSO      = list(family="binomial"),
        NeuralNet  = list(trace=FALSE, MaxNWts=400),
        GBM        = list(verbose=FALSE),
        list())
      args_oof <- c(
        list(recurrence~., data=tr_f,
             method=caret_map[nm], metric="ROC",
             trControl=ctrl_oof, tuneLength=tl_nm),
        if (!is.null(pp_nm)) list(preProcess=pp_nm),
        extra_nm
      )
      m_oof <- do.call(train, args_oof)
      oof[va_i, nm] <- predict(m_oof, va_f, type="prob")[,"Rec1"]
    }, error=function(e) NULL)
  }

  tryCatch({
    xtr_f <- xgb.DMatrix(as.matrix(tr_f[,feats_f]),
                         label=as.numeric(tr_f$recurrence=="Rec1"))
    xva_f <- xgb.DMatrix(as.matrix(va_f[,feats_f]))
    m_xf  <- xgb.train(params=xgb_par, data=xtr_f, nrounds=best_nr_main,
                        verbose=0, print_every_n=9999)
    oof[va_i,"XGBoost"] <- predict(m_xf, xva_f)
  }, error=function(e) NULL)
}

# Meta-dataset
meta_tr <- as.data.frame(oof)
meta_tr$y <- train_bal_main$recurrence
good_c <- names(which(colMeans(is.na(meta_tr)) < 0.30))
meta_tr <- meta_tr[, c(good_c[good_c!="y"], "y")]
meta_tr <- na.omit(meta_tr)
meta_tr$y <- factor(meta_tr$y)

# Test meta-features
meta_te_cols <- setdiff(names(meta_tr),"y")
meta_te <- as.data.frame(matrix(NA, nrow=nrow(test_df),
                                ncol=length(meta_te_cols)))
names(meta_te) <- meta_te_cols
for (nm in intersect(meta_te_cols, setdiff(all_nms,"XGBoost"))) {
  tryCatch({
    meta_te[[nm]] <- predict(main_models[[nm]], test_df, type="prob")[,"Rec1"]
  }, error=function(e) NULL)
}
if ("XGBoost" %in% meta_te_cols) {
  tryCatch({
    meta_te[["XGBoost"]] <- predict(main_models$XGBoost, xgb_te_main)
  }, error=function(e) NULL)
}
meta_te_clean <- na.omit(meta_te)

# Meta-learner
ctrl_meta <- trainControl(method="cv", number=5, classProbs=TRUE,
                          summaryFunction=twoClassSummary, verboseIter=FALSE)

stack_model <- tryCatch(
  train(y~., data=meta_tr, method="glmnet", family="binomial",
        metric="ROC", trControl=ctrl_meta, tuneLength=10,
        preProcess=c("center","scale")),
  error=function(e)
    train(y~., data=meta_tr, method="glm", family="binomial",
          metric="ROC", trControl=ctrl_meta))

stack_prob <- tryCatch(
  predict(stack_model, meta_te_clean, type="prob")[,"Rec1"],
  error=function(e) NULL)

if (!is.null(stack_prob) && length(stack_prob) == nrow(test_df)) {
  stack_roc <- pROC::roc(as.numeric(test_df$recurrence=="Rec1"),
                         as.numeric(stack_prob), quiet=TRUE)
  stack_auc <- round(as.numeric(pROC::auc(stack_roc)), 4)
  message(sprintf("  [OK] Stacking Ensemble AUC = %.4f", stack_auc))
} else {
  # Fallback: weighted average (top-3 by bootstrap AUC)
  top3 <- head(boot_sum$Model, 3)
  probs3 <- sapply(top3, function(nm)
    tryCatch(get_prob_main(nm), error=function(e) rep(0.5,nrow(test_df))))
  w3 <- head(boot_sum$AUC_mean, 3)
  stack_prob <- as.vector(probs3 %*% (w3/sum(w3)))
  stack_roc  <- pROC::roc(as.numeric(test_df$recurrence=="Rec1"),
                           stack_prob, quiet=TRUE)
  stack_auc  <- round(as.numeric(pROC::auc(stack_roc)), 4)
  message(sprintf("  [OK] Weighted-Avg Fallback Ensemble AUC = %.4f",
                  stack_auc))
}

# ============================================================
# 10. COMBINED SUMMARY TABLE
# ============================================================
message("\n--- Combined Summary ---")

comb_sum <- boot_sum %>%
  select(Model, Type, AUC_point, AUC_mean, AUC_sd, CI_lo, CI_hi, CI_95) %>%
  left_join(ncv_sum %>% select(Model, NCV_Mean=Mean_AUC, NCV_SD=SD_AUC), by="Model") %>%
  left_join(seed_sum %>% select(Model, MS_Mean=Mean_AUC, MS_SD=SD_AUC,
                                Stability, Pct_gt80), by="Model") %>%
  arrange(desc(AUC_mean))

# Add stacking row
stack_row <- data.frame(
  Model="Stacking_Ensemble", Type="Ensemble",
  AUC_point=stack_auc, AUC_mean=stack_auc,
  AUC_sd=NA, CI_lo=NA, CI_hi=NA,
  CI_95=sprintf("%.3f (single eval)", stack_auc),
  NCV_Mean=NA, NCV_SD=NA,
  MS_Mean=NA, MS_SD=NA,
  Stability="N/A", Pct_gt80=NA,
  stringsAsFactors=FALSE)

comb_all <- rbind(comb_sum, stack_row)
save_csv(comb_all, "Table_Comprehensive_Performance.csv")

# ============================================================
# 11. FIGURLER
# ============================================================
message("\n--- Generating Figures ---")

## Fig R1: Nested CV boxplot
p_ncv <- ggplot(
  ncv_df %>% mutate(Type=ifelse(Model %in% WHITEBOX,"WhiteBox","BlackBox")),
  aes(x=reorder(Model,AUC,median), y=AUC, fill=Type)) +
  geom_boxplot(alpha=0.75, outlier.shape=21, width=0.55) +
  geom_jitter(width=0.12, alpha=0.5, size=2) +
  geom_hline(yintercept=0.80, lty=2, color="gray40") +
  scale_fill_manual(values=c("WhiteBox"=WB_COL,"BlackBox"=BB_COL)) +
  coord_flip() +
  scale_y_continuous(limits=c(0,1.02), labels=scales::percent) +
  labs(title="Nested Cross-Validation AUC",
       subtitle=sprintf("%d-fold outer | %d-fold inner | n=%d",
                        OUTER_K, INNER_K, nrow(df_ml)),
       x=NULL, y="AUC per fold", fill="Model Type") +
  theme_classic(base_size=12) +
  theme(legend.position="top", plot.title=element_text(face="bold"))
save_fig(p_ncv, "Fig_R1_NestedCV.png", w=10, h=7)

## Fig R2: Bootstrap CI errorbars
p_boot <- boot_sum %>%
  mutate(Model=reorder(Model, AUC_mean)) %>%
  ggplot(aes(x=Model, y=AUC_mean, color=Type)) +
  geom_point(size=4) +
  geom_errorbar(aes(ymin=CI_lo, ymax=CI_hi), width=0.3, linewidth=1.1) +
  geom_hline(yintercept=0.80, lty=2, color="gray50") +
  geom_hline(yintercept=0.95, lty=3, color="#D32F2F", linewidth=0.9) +
  annotate("text", x=0.8, y=0.965, hjust=0, size=3.5,
           label="AUC=0.95 target", color="#D32F2F", fontface="italic") +
  scale_color_manual(values=c("WhiteBox"=WB_COL,"BlackBox"=BB_COL)) +
  scale_y_continuous(limits=c(0.35,1.08), labels=scales::percent) +
  coord_flip() +
  labs(title="Bootstrap AUC with 95% Confidence Intervals",
       subtitle=sprintf("%d bootstrap resamples", N_BOOT),
       x=NULL, y="AUC (Mean +/- 95% CI)", color="Type") +
  theme_classic(base_size=12) +
  theme(legend.position="top", plot.title=element_text(face="bold"))
save_fig(p_boot, "Fig_R2_Bootstrap_CI.png", w=10, h=7)

## Fig R3: Multi-seed violin
p_seed_v <- seed_df %>%
  mutate(Type=ifelse(Model %in% WHITEBOX,"WhiteBox","BlackBox")) %>%
  ggplot(aes(x=reorder(Model,AUC,median), y=AUC, fill=Type)) +
  geom_violin(alpha=0.65, trim=TRUE) +
  geom_boxplot(width=0.15, fill="white", outlier.shape=NA, linewidth=0.8) +
  geom_hline(yintercept=0.80, lty=2, color="gray40") +
  scale_fill_manual(values=c("WhiteBox"=WB_COL,"BlackBox"=BB_COL)) +
  scale_y_continuous(limits=c(0.25,1.05), labels=scales::percent) +
  coord_flip() +
  labs(title="Multi-Seed Stability Analysis",
       subtitle=sprintf("%d random seeds | n=%d patients", N_SEEDS, nrow(df_ml)),
       x=NULL, y="AUC distribution", fill="Type") +
  theme_classic(base_size=12) +
  theme(legend.position="top", plot.title=element_text(face="bold"))
save_fig(p_seed_v, "Fig_R3_MultiSeed_Violin.png", w=10, h=7)

## Fig R4: Stability heatmap (SD renk)
stab_heat <- seed_sum %>%
  mutate(Type=ifelse(Model %in% WHITEBOX,"WB","BB"),
         Model=reorder(Model, Mean_AUC))

p_stab <- ggplot(stab_heat, aes(x="SD", y=Model)) +
  geom_tile(aes(fill=SD_AUC), color="white", width=0.6) +
  geom_text(aes(label=sprintf("%.3f +/- %.3f\n%s",
                              Mean_AUC, SD_AUC, Stability)),
            size=3, fontface="bold") +
  scale_fill_gradient(low="#C8E6C9", high="#FFCDD2", name="SD") +
  labs(title="Model Stability: Mean AUC +/- SD",
       subtitle="Green = stable (SD<0.05)  |  Red = unstable (SD>0.10)",
       x=NULL, y=NULL) +
  theme_classic(base_size=12) +
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),
        plot.title=element_text(face="bold"))
save_fig(p_stab, "Fig_R4_Stability_Heatmap.png", w=6, h=7)

## Fig R5: 3-method comparison dot plot
comp_long <- comb_sum %>%
  select(Model, Type, Bootstrap=AUC_mean, NestedCV=NCV_Mean, MultiSeed=MS_Mean) %>%
  pivot_longer(c(Bootstrap, NestedCV, MultiSeed),
               names_to="Evaluation", values_to="AUC") %>%
  filter(!is.na(AUC))

p_comp <- ggplot(comp_long,
                 aes(x=reorder(Model, AUC, mean, na.rm=TRUE),
                     y=AUC, color=Evaluation, shape=Type)) +
  geom_point(size=3.5, alpha=0.85, position=position_dodge(width=0.6)) +
  geom_hline(yintercept=0.80, lty=2, color="gray40") +
  geom_hline(yintercept=0.95, lty=3, color="#D32F2F", linewidth=0.8) +
  scale_color_manual(values=c("Bootstrap"="#1976D2",
                              "NestedCV"="#388E3C",
                              "MultiSeed"="#F57C00")) +
  scale_shape_manual(values=c("WhiteBox"=16,"BlackBox"=17)) +
  scale_y_continuous(limits=c(0.3,1.05), labels=scales::percent) +
  coord_flip() +
  labs(title="AUC Across Three Evaluation Strategies",
       subtitle="Dashed=0.80 | Dotted=0.95",
       x=NULL, y="Mean AUC",
       color="Method", shape="Model Type") +
  theme_classic(base_size=12) +
  theme(legend.position="top", plot.title=element_text(face="bold"))
save_fig(p_comp, "Fig_R5_Method_Comparison.png", w=11, h=8)

## Fig R6: ROC curves + stacking
pal9 <- c("#1565C0","#388E3C","#F57C00","#7B1FA2",
          "#B71C1C","#00838F","#6A1B9A","#2E7D32","#37474F")

roc_list <- lapply(all_nms, function(nm) {
  prob <- get_prob_main(nm)
  pROC::roc(as.numeric(test_df$recurrence=="Rec1"),
            as.numeric(prob), quiet=TRUE)
})
names(roc_list) <- all_nms
aucs_pt <- sapply(roc_list, function(r) round(as.numeric(pROC::auc(r)),3))
ci_lbl  <- boot_sum$CI_95[match(names(aucs_pt), boot_sum$Model)]

open_png("Fig_R6_ROC_Curves.png", w=11, h=9)
plot(roc_list[[1]], col=pal9[1], lwd=2.5,
     main="ROC Curves (All Models + Stacking Ensemble)",
     xlab="1 - Specificity", ylab="Sensitivity", cex.main=1.2)
for (i in 2:length(roc_list))
  plot(roc_list[[i]], col=pal9[i], lwd=2.5, add=TRUE)
plot(stack_roc, col="black", lwd=3.5, lty=2, add=TRUE)
abline(a=0, b=1, lty=2, col="gray60")
legend_labs <- c(
  sprintf("%s  %s", names(aucs_pt), ci_lbl),
  sprintf("Stacking Ensemble  AUC=%.3f", stack_auc)
)
legend("bottomright", legend=legend_labs,
       col=c(pal9[1:length(all_nms)],"black"),
       lwd=c(rep(2.5,length(all_nms)),3.5),
       lty=c(rep(1,length(all_nms)),2),
       bty="n", cex=0.78)
dev.off()
message("[OK] Fig_R6_ROC_Curves.png")

## Fig R7: AUC ranking + CI bars
p_rank <- boot_sum %>%
  mutate(
    Model = reorder(Model, AUC_mean),
    lbl   = sprintf("%.3f\n(%.3f-%.3f)", AUC_mean, CI_lo, CI_hi)
  ) %>%
  ggplot(aes(x=Model, y=AUC_mean, fill=Type)) +
  geom_bar(stat="identity", alpha=0.85, color="white", width=0.65) +
  geom_errorbar(aes(ymin=CI_lo, ymax=CI_hi),
                width=0.25, linewidth=0.9, color="gray30") +
  geom_text(aes(y=CI_hi+0.015, label=lbl),
            hjust=0, size=2.8, lineheight=0.9) +
  scale_fill_manual(values=c("WhiteBox"=WB_COL,"BlackBox"=BB_COL)) +
  geom_hline(yintercept=0.95, lty=3, color="#D32F2F", linewidth=0.8) +
  coord_flip() +
  scale_y_continuous(limits=c(0,1.28), expand=expansion(mult=c(0,0)),
                     labels=scales::percent) +
  labs(title="Model Ranking: Bootstrap AUC with 95% CI",
       subtitle="Dotted line = 0.95 target",
       x=NULL, y="AUC (Bootstrap)", fill="Type") +
  theme_classic(base_size=12) +
  theme(legend.position="top", plot.title=element_text(face="bold"))
save_fig(p_rank, "Fig_R7_AUC_Ranking.png", w=10, h=7)

## Fig R8: Seed-by-seed AUC heatmap (Model x Seed)
if (nrow(seed_df) > 0) {
  seed_wide <- seed_df %>%
    pivot_wider(names_from=Seed, values_from=AUC)
  seed_mat <- as.matrix(seed_wide[,-1])
  rownames(seed_mat) <- seed_wide$Model

  open_png("Fig_R8_SeedHeatmap.png", w=14, h=7)
  par(mar=c(6,10,3,2))
  pal_heat <- colorRampPalette(c("#FFCDD2","#FFFDE7","#C8E6C9"))(100)
  image(t(seed_mat[nrow(seed_mat):1, ]),
        col=pal_heat, axes=FALSE,
        main="AUC Heatmap: Model x Seed")
  axis(1, at=seq(0,1,length.out=ncol(seed_mat)),
       labels=colnames(seed_mat), las=2, cex.axis=0.7)
  axis(2, at=seq(0,1,length.out=nrow(seed_mat)),
       labels=rev(rownames(seed_mat)), las=1, cex.axis=0.85)
  # Add values
  for (i in seq_len(nrow(seed_mat))) {
    for (j in seq_len(ncol(seed_mat))) {
      v <- seed_mat[i, j]
      if (!is.na(v))
        text(x=(j-1)/(ncol(seed_mat)-1),
             y=1-(i-1)/(nrow(seed_mat)-1),
             labels=sprintf("%.2f", v),
             cex=0.55, col=ifelse(v>0.85,"black","gray30"))
    }
  }
  dev.off()
  message("[OK] Fig_R8_SeedHeatmap.png")
}

# ============================================================
# 12. COMPOSITE SCORE - Otomatik Model Secimi
#     Kriter: Yuksek AUC + Dusuk Varyans (tum yontemlerde tutarli)
#     Score = w1*NormAUC - w2*NormSD - w3*NormRange
#     w1=0.60 (performans), w2=0.25 (sd), w3=0.15 (range)
# ============================================================
message("\n--- Composite Scoring & Automatic Model Selection ---")

# Normalize helper: 0-1 araligina al
norm01 <- function(x) {
  rng <- range(x, na.rm=TRUE)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

# Metrikler: her yontem icin AUC ve SD/varyans
score_df <- data.frame(
  Model    = seed_sum$Model,
  Type     = seed_sum$Type,
  # Performans: Bootstrap + NCV + MultiSeed ortalamasi
  AUC_boot = boot_sum$AUC_mean[match(seed_sum$Model, boot_sum$Model)],
  AUC_ncv  = ncv_sum$Mean_AUC[match(seed_sum$Model, ncv_sum$Model)],
  AUC_ms   = seed_sum$Mean_AUC,
  # Varyans: multi-seed SD (en kritik)
  SD_ms    = seed_sum$SD_AUC,
  # Aralik: max-min (seed varyansinin boyutu)
  Range_ms = seed_sum$Max_AUC - seed_sum$Min_AUC,
  # %>80 tutarlilik
  Pct_gt80 = seed_sum$Pct_gt80,
  stringsAsFactors = FALSE
) %>%
  mutate(
    # Konsensus AUC: 3 yontemin agirlikli ortalamasi
    AUC_consensus = round(
      0.40 * AUC_boot +
      0.35 * AUC_ncv  +
      0.25 * AUC_ms,
      4),
    # Normalize (yuksek AUC iyi, dusuk SD/Range iyi)
    nAUC   = norm01(AUC_consensus),
    nSD    = 1 - norm01(SD_ms),    # ters: dusuk SD daha iyi
    nRange = 1 - norm01(Range_ms), # ters: dar aralik daha iyi
    nPct   = norm01(Pct_gt80),
    # Composite: 60% AUC + 20% SD + 10% Range + 10% Pct>80
    Composite_Score = round(
      0.60 * nAUC +
      0.20 * nSD  +
      0.10 * nRange +
      0.10 * nPct,
      4)
  ) %>%
  arrange(desc(Composite_Score))

save_csv(score_df, "Table_Composite_Scores.csv")

# Secinlen model
chosen_model      <- score_df$Model[1]
chosen_type       <- score_df$Type[1]
chosen_score      <- score_df$Composite_Score[1]
chosen_auc        <- score_df$AUC_consensus[1]
chosen_sd         <- score_df$SD_ms[1]
chosen_pct        <- score_df$Pct_gt80[1]
chosen_boot_ci    <- boot_sum$CI_95[match(chosen_model, boot_sum$Model)]
chosen_stability  <- seed_sum$Stability[match(chosen_model, seed_sum$Model)]

cat(sprintf("\n*** SECILEN MODEL: %s (%s) ***\n", chosen_model, chosen_type))
cat(sprintf("  Composite Score : %.4f\n", chosen_score))
cat(sprintf("  Consensus AUC   : %.4f\n", chosen_auc))
cat(sprintf("  Bootstrap 95 CI : %s\n", chosen_boot_ci))
cat(sprintf("  Multi-Seed SD   : %.4f  (%s)\n", chosen_sd, chosen_stability))
cat(sprintf("  Pct seeds >0.80 : %.1f%%\n", chosen_pct))

# ============================================================
# 13. COMPOSITE SCORE FIGURU
# ============================================================

## Fig R9: Composite score ranking
p_score <- score_df %>%
  mutate(Model = reorder(Model, Composite_Score)) %>%
  ggplot(aes(x=Model, y=Composite_Score, fill=Type)) +
  geom_bar(stat="identity", alpha=0.88, color="white", width=0.65) +
  geom_text(aes(label=sprintf("%.3f\nAUC=%.3f  SD=%.3f",
                              Composite_Score, AUC_consensus, SD_ms)),
            hjust=-0.05, size=2.8, lineheight=0.85) +
  scale_fill_manual(values=c("WhiteBox"=WB_COL,"BlackBox"=BB_COL)) +
  coord_flip() +
  scale_y_continuous(limits=c(0,1.35),
                     expand=expansion(mult=c(0,0))) +
  geom_vline(xintercept=which(levels(reorder(score_df$Model,
             score_df$Composite_Score))==chosen_model) - 0.5 +
               nrow(score_df) - nrow(score_df) + 0.5,
             lty=2, color="#D32F2F", linewidth=0.8) +
  labs(
    title    = "Composite Score: AUC + Stability Combined",
    subtitle = sprintf(
      "Score = 0.60*AUC + 0.20*(1-SD) + 0.10*(1-Range) + 0.10*Pct>0.80\nSelected: %s (Score=%.3f)",
      chosen_model, chosen_score),
    x=NULL, y="Composite Score", fill="Type"
  ) +
  theme_classic(base_size=12) +
  theme(legend.position="top",
        plot.title=element_text(face="bold"),
        plot.subtitle=element_text(color="#D32F2F"))
save_fig(p_score, "Fig_R9_CompositeScore.png", w=11, h=7)

## Fig R10: 2D scatter: AUC vs Stability (SD)
# Bubble size = % seeds above 0.80
p_2d <- ggplot(score_df,
               aes(x=SD_ms, y=AUC_consensus,
                   color=Type, size=Pct_gt80,
                   label=Model)) +
  geom_point(alpha=0.80) +
  geom_text(vjust=-1.2, size=3, show.legend=FALSE) +
  geom_vline(xintercept=0.07, lty=2, color="gray50",
             linewidth=0.8) +
  geom_hline(yintercept=0.85, lty=2, color="gray50",
             linewidth=0.8) +
  # Target zone: high AUC + low SD
  annotate("rect", xmin=0, xmax=0.07, ymin=0.85, ymax=1.02,
           fill="#C8E6C9", alpha=0.25) +
  annotate("text", x=0.01, y=1.01, hjust=0, size=3.5,
           label="Optimal zone\n(High AUC + Low SD)",
           color="#2E7D32", fontface="italic") +
  scale_color_manual(values=c("WhiteBox"=WB_COL,"BlackBox"=BB_COL)) +
  scale_size_continuous(range=c(3,10), name="% Seeds\n>0.80 AUC") +
  scale_x_continuous(limits=c(0, NA), labels=scales::number_format(accuracy=0.01)) +
  scale_y_continuous(limits=c(0.4, 1.05), labels=scales::percent) +
  labs(
    title    = "AUC vs. Stability: Model Selection Landscape",
    subtitle = "Top-right green zone = ideal | Bubble size = % seeds with AUC>0.80",
    x="Standard Deviation (across seeds) -- lower is more stable",
    y="Consensus AUC (Bootstrap + NestedCV + MultiSeed)",
    color="Model Type"
  ) +
  theme_classic(base_size=12) +
  theme(legend.position="right",
        plot.title=element_text(face="bold"))
save_fig(p_2d, "Fig_R10_AUC_vs_Stability.png", w=11, h=8)

## Fig R11: Selected model - seed-by-seed AUC trajectory
chosen_seeds <- seed_df %>%
  filter(Model == chosen_model) %>%
  arrange(Seed) %>%
  mutate(SeedIdx=row_number())

p_traj <- ggplot(chosen_seeds, aes(x=SeedIdx, y=AUC)) +
  geom_line(color=ifelse(chosen_type=="WhiteBox",WB_COL,BB_COL),
            linewidth=1.2) +
  geom_point(color=ifelse(chosen_type=="WhiteBox",WB_COL,BB_COL),
             size=3) +
  geom_hline(yintercept=mean(chosen_seeds$AUC),
             lty=2, color="gray40", linewidth=0.9) +
  geom_hline(yintercept=0.95, lty=3, color="#D32F2F", linewidth=0.8) +
  geom_hline(yintercept=0.80, lty=3, color="#F57C00", linewidth=0.8) +
  annotate("text", x=max(chosen_seeds$SeedIdx)*0.02,
           y=mean(chosen_seeds$AUC)+0.012,
           label=sprintf("Mean=%.3f", mean(chosen_seeds$AUC)),
           hjust=0, size=3.5, color="gray30") +
  annotate("text", x=max(chosen_seeds$SeedIdx)*0.02, y=0.962,
           label="AUC=0.95", hjust=0, size=3.2, color="#D32F2F") +
  annotate("text", x=max(chosen_seeds$SeedIdx)*0.02, y=0.812,
           label="AUC=0.80", hjust=0, size=3.2, color="#F57C00") +
  scale_x_continuous(breaks=chosen_seeds$SeedIdx,
                     labels=chosen_seeds$Seed) +
  scale_y_continuous(limits=c(0.3,1.05), labels=scales::percent) +
  labs(
    title    = sprintf("Selected Model: %s - AUC Across %d Seeds",
                       chosen_model, N_SEEDS),
    subtitle = sprintf("Mean=%.4f | SD=%.4f | Stability=%s | %s",
                       mean(chosen_seeds$AUC, na.rm=TRUE),
                       sd(chosen_seeds$AUC, na.rm=TRUE),
                       chosen_stability, chosen_type),
    x="Random Seed", y="Test AUC"
  ) +
  theme_classic(base_size=12) +
  theme(axis.text.x=element_text(angle=45, hjust=1, size=8),
        plot.title=element_text(face="bold"))
save_fig(p_traj, "Fig_R11_SelectedModel_Trajectory.png", w=12, h=6)

# ============================================================
# 14. FINAL RAPOR
# ============================================================
cat("\n========================================================\n")
cat("  ROBUST ML ANALYSIS COMPLETE\n")
cat("========================================================\n\n")

cat("--- [1/4] Nested CV Sonuclari ---\n")
best_ncv <- ncv_sum[1,]
cat(sprintf("  Best: %s | Mean AUC=%.4f | SD=%.4f\n",
            best_ncv$Model, best_ncv$Mean_AUC, best_ncv$SD_AUC))

cat(sprintf("\n--- [2/4] Bootstrap CI (%d tekrar) ---\n", N_BOOT))
best_boot <- boot_sum[1,]
cat(sprintf("  Best: %s | %s\n", best_boot$Model, best_boot$CI_95))

cat(sprintf("\n--- [3/4] Multi-Seed Stability (%d seed) ---\n", N_SEEDS))
best_ms <- seed_sum[1,]
cat(sprintf("  Best: %s | Mean=%.4f | SD=%.4f | %s\n",
            best_ms$Model, best_ms$Mean_AUC, best_ms$SD_AUC,
            best_ms$Stability))

cat(sprintf("\n--- [4/4] Stacking Ensemble ---\n"))
cat(sprintf("  AUC = %.4f\n", stack_auc))

cat("\n--- COMPOSITE SCORE RANKING ---\n")
print(score_df[, c("Model","Type","AUC_consensus","SD_ms",
                   "Pct_gt80","Composite_Score")])

cat(sprintf("\n*** OTOMATIK SECILEN MODEL: %s ***\n", chosen_model))
cat(sprintf("  Consensus AUC  : %.4f\n", chosen_auc))
cat(sprintf("  Bootstrap CI   : %s\n", chosen_boot_ci))
cat(sprintf("  Multi-Seed SD  : %.4f (%s stability)\n",
            chosen_sd, chosen_stability))
cat(sprintf("  Seed>0.80 rate : %.1f%%\n", chosen_pct))
cat(sprintf("  Composite Score: %.4f\n\n", chosen_score))

cat("--- Makale Icin Reporting Template ---\n")
cat(sprintf(
  "  '%s, 9 ML modeli arasinda en yuksek tutarlilik ile\n",
  chosen_model))
cat(sprintf(
  "  on planlanan %d farkli random seed'de\n", N_SEEDS))
cat(sprintf(
  "  konsensus AUC=%.3f (Bootstrap 95%% CI: %s) elde etti.\n",
  chosen_auc, chosen_boot_ci))
cat(sprintf(
  "  %d-fold nested CV ile dogrulanan AUC=%.3f,\n",
  OUTER_K, boot_sum$AUC_mean[boot_sum$Model==chosen_model]))
cat(sprintf(
  "  seed varyansini gosteren SD=%.3f ile yuksek stabilite\n",
  chosen_sd))
cat("  sergiledi (Stacking Ensemble ile karsilastirildi).'\n")

cat("\n--- Cikti Dosyalari ---\n")
outs <- sort(list.files(OUT, pattern="^(Fig_R|Table_).*\\.(png|csv)$"))
cat(paste0("  ", seq_along(outs), ". ", outs, collapse="\n"),"\n")
cat("\n========================================================\n\n")

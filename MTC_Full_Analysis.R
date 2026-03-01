cat("\n")
cat("========================================================\n")
cat("  MTC COMPREHENSIVE ANALYSIS - Starting...\n")
cat("========================================================\n\n")

# ============================================================
# 1. PACKAGES
# ============================================================
pkgs <- c(
  "tidyverse", "tableone", "survival", "survminer",
  "caret", "randomForest", "glmnet", "pROC",
  "xgboost", "e1071", "ROSE",
  "naivebayes", "rpart", "rpart.plot", "nnet", "gbm",
  "DALEX", "DALEXtra",
  "corrplot", "ggplot2", "ggpubr", "cowplot",
  "scales", "viridis", "gridExtra"
)

installed <- rownames(installed.packages())
to_install <- pkgs[!pkgs %in% installed]
if (length(to_install) > 0) {
  message("Installing missing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(
  invisible(lapply(pkgs, library, character.only = TRUE))
)
message("[OK] All packages loaded")

# ============================================================
# 2. GLOBAL SETTINGS
# ============================================================
set.seed(42)
DPI    <- 600
FIG_W  <- 10
FIG_H  <- 8
OUT    <- getwd()

# Helper: save ggplot at 600 DPI
save_fig <- function(plot_obj, filename, w = FIG_W, h = FIG_H) {
  ggsave(file.path(OUT, filename), plot_obj,
         width = w, height = h, dpi = DPI, units = "in")
  message("[OK] ", filename)
}

# Helper: save base-R plot at 600 DPI
open_png <- function(filename, w = FIG_W, h = FIG_H) {
  png(file.path(OUT, filename), width = w, height = h,
      units = "in", res = DPI)
}

# Helper: save CSV
save_csv <- function(df, filename) {
  write.csv(df, file.path(OUT, filename), row.names = FALSE)
  message("[OK] ", filename)
}

# Model type classification
WHITEBOX <- c("LASSO", "CART", "NaiveBayes", "KNN")
BLACKBOX <- c("RandomForest", "SVM", "NeuralNet", "GBM", "XGBoost")
WB_COL   <- "#1565C0"
BB_COL   <- "#B71C1C"

# ============================================================
# 3. DATA LOADING & CLEANING
# ============================================================
message("\n--- [1/15] Data Loading & Cleaning ---")

if (!file.exists("DATA_2.csv"))
  stop("DATA_2.csv not found in working directory: ", getwd())

df_raw <- read.csv("DATA_2.csv", stringsAsFactors = FALSE,
                   na.strings = c("", "NA"))
df_raw <- df_raw[, !duplicated(names(df_raw))]

num_force <- c("age_at_diagnosis", "follow_up_year", "tumor_size_mm",
               "usg_size_mm", "calcitonin_preop", "calcitonin_postop",
               "cea_preop", "cea_postop", "recurrence_time_months")

factor_vars <- c(
  "sex", "disease_type", "ret_mutation", "pheochromocytoma",
  "tumor_side", "multifocal", "lymph_node_invasion",
  "capsular_invasion", "soft_tissue_invasion",
  "metastasis_at_diagnosis", "mediastinal_metastasis",
  "lung_metastasis_at_diagnosis", "metastasis_followup",
  "lung_metastasis_followup", "liver_metastasis", "bone_metastasis",
  "stage", "lymph_node_metastasis", "bone_metastasis_present",
  "lung_metastasis_present", "liver_metastasis_present",
  "adrenal_metastasis", "recurrence", "tt_cnd_snd", "tt_cnd",
  "total_thyroidectomy", "completion_thyroidectomy",
  "hemithyroidectomy", "disease_status", "vital_status",
  "alive_no_disease", "alive_with_disease", "alive_unknown_status",
  "death_mtc", "tki_therapy", "radiotherapy", "lutetium_therapy",
  "chemotherapy", "mibg_therapy"
)

df <- df_raw %>%
  mutate(across(all_of(num_force), ~ as.numeric(as.character(.)))) %>%
  mutate(across(all_of(intersect(factor_vars, names(.))), as.factor)) %>%
  mutate(
    sex_label          = factor(sex, levels = c(0,1),
                                labels = c("Female", "Male")),
    disease_type_label = factor(disease_type, levels = c(0,1),
                                labels = c("Sporadic", "Hereditary")),
    stage_label        = factor(stage, levels = c(1,2,3,4),
                                labels = c("I", "II", "III", "IV")),
    outcome_death      = as.numeric(as.character(death_mtc)),
    recurrence_num     = as.numeric(as.character(recurrence)),
    time_surv          = as.numeric(as.character(follow_up_year)),
    event_surv         = as.numeric(as.character(death_mtc))
  )

cat(sprintf("Dataset: %d patients x %d variables\n", nrow(df), ncol(df)))
cat("Missing data (key vars):\n")
print(colSums(is.na(df[, c("calcitonin_preop", "calcitonin_postop",
                            "cea_preop", "tumor_size_mm",
                            "follow_up_year", "recurrence")])))

# ============================================================
# 4. TABLE 1 - BASELINE CHARACTERISTICS
# ============================================================
message("\n--- [2/15] Table 1: Baseline Characteristics ---")

t1_vars <- c("age_at_diagnosis", "sex_label", "disease_type_label",
             "ret_mutation", "pheochromocytoma", "tumor_size_mm",
             "stage_label", "multifocal", "lymph_node_invasion",
             "capsular_invasion", "soft_tissue_invasion",
             "metastasis_at_diagnosis", "calcitonin_preop", "cea_preop",
             "recurrence", "tki_therapy", "follow_up_year")

t1_cat <- c("sex_label", "disease_type_label", "ret_mutation",
            "pheochromocytoma", "stage_label", "multifocal",
            "lymph_node_invasion", "capsular_invasion",
            "soft_tissue_invasion", "metastasis_at_diagnosis",
            "recurrence", "tki_therapy")

tab1 <- CreateTableOne(
  vars       = t1_vars,
  factorVars = t1_cat,
  strata     = "disease_type_label",
  data       = df,
  addOverall = TRUE
)
tab1_df <- as.data.frame(
  print(tab1, quote = FALSE, noSpaces = TRUE, printToggle = FALSE,
        nonnormal = c("calcitonin_preop", "cea_preop", "tumor_size_mm"))
)
tab1_df$Variable <- rownames(tab1_df)
tab1_df <- tab1_df[, c("Variable", setdiff(names(tab1_df), "Variable"))]
save_csv(tab1_df, "Table1_Baseline_Characteristics.csv")

# ============================================================
# 5. EXPLORATORY FIGURES
# ============================================================
message("\n--- [3/15] Exploratory Figures ---")

## Fig1A: Age distribution by disease type
p1a <- ggplot(df, aes(x = age_at_diagnosis, fill = disease_type_label)) +
  geom_density(alpha = 0.65, color = "white") +
  scale_fill_manual(values = c("#1976D2", "#D32F2F")) +
  labs(title = "A. Age Distribution by Disease Type",
       x = "Age at Diagnosis (years)", y = "Density",
       fill = "Disease Type") +
  theme_classic(base_size = 12)

## Fig1B: Tumor size by stage
p1b <- ggplot(df %>% filter(!is.na(stage_label)),
              aes(x = stage_label, y = tumor_size_mm, fill = stage_label)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 21, outlier.fill = "red") +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "B. Tumor Size by AJCC Stage",
       x = "Stage", y = "Tumor Size (mm)") +
  theme_classic(base_size = 12) +
  theme(legend.position = "none")

## Fig1C: Calcitonin pre vs post (log scale)
df_cal_long <- df %>%
  select(calcitonin_preop, calcitonin_postop, disease_type_label) %>%
  pivot_longer(c(calcitonin_preop, calcitonin_postop),
               names_to = "Timepoint", values_to = "Calcitonin") %>%
  mutate(Timepoint = recode(Timepoint,
                            calcitonin_preop  = "Pre-operative",
                            calcitonin_postop = "Post-operative")) %>%
  filter(!is.na(Calcitonin))

p1c <- ggplot(df_cal_long, aes(x = Timepoint,
                                y = log10(Calcitonin + 1),
                                fill = Timepoint)) +
  geom_boxplot(alpha = 0.8) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  scale_fill_manual(values = c("#388E3C", "#F57C00")) +
  stat_compare_means(method = "wilcox.test",
                     label = "p.format", size = 4) +
  labs(title = "C. Calcitonin (log10+1 pg/mL)",
       x = "", y = "log10(Calcitonin + 1)") +
  theme_classic(base_size = 12) +
  theme(legend.position = "none")

## Fig1D: Treatment frequency
treat_df <- df %>%
  summarise(
    "TKI Therapy"      = sum(tki_therapy == 1, na.rm = TRUE),
    "Radiotherapy"     = sum(radiotherapy == 1, na.rm = TRUE),
    "Lutetium"         = sum(lutetium_therapy == 1, na.rm = TRUE),
    "Chemotherapy"     = sum(chemotherapy == 1, na.rm = TRUE),
    "MIBG"             = sum(mibg_therapy == 1, na.rm = TRUE)
  ) %>%
  pivot_longer(everything(), names_to = "Treatment", values_to = "Count")

p1d <- ggplot(treat_df,
              aes(x = reorder(Treatment, -Count), y = Count,
                  fill = Treatment)) +
  geom_bar(stat = "identity", color = "white", alpha = 0.85) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "D. Treatment Frequency",
       x = "", y = "Number of Patients") +
  theme_classic(base_size = 12) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 30, hjust = 1))

fig1 <- plot_grid(p1a, p1b, p1c, p1d, ncol = 2, labels = NULL)
save_fig(fig1, "Fig1_Exploratory_Overview.png")

# ============================================================
# 6. CORRELATION MATRIX
# ============================================================
message("\n--- [4/15] Correlation Matrix ---")

num_vars <- c("age_at_diagnosis", "tumor_size_mm", "usg_size_mm",
              "calcitonin_preop", "calcitonin_postop",
              "cea_preop", "cea_postop",
              "follow_up_year", "recurrence_time_months")

cor_mat <- cor(
  df[, num_vars] %>%
    mutate(across(everything(), ~ as.numeric(as.character(.)))),
  use    = "pairwise.complete.obs",
  method = "spearman"
)
colnames(cor_mat) <- rownames(cor_mat) <-
  c("Age", "Tumor\nSize", "USG\nSize",
    "Cal\nPreop", "Cal\nPostop",
    "CEA\nPreop", "CEA\nPostop",
    "Follow-up", "Recur.\nTime")

open_png("Fig2_Correlation_Matrix.png")
corrplot(cor_mat, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.85,
         addCoef.col = "black", number.cex = 0.7,
         col = colorRampPalette(c("#D32F2F", "white", "#1976D2"))(200),
         title = "Spearman Correlation - Continuous Variables",
         mar   = c(0, 0, 2, 0))
dev.off()
message("[OK] Fig2_Correlation_Matrix.png")

# ============================================================
# 7. SURVIVAL ANALYSIS
# ============================================================
message("\n--- [5/15] Survival Analysis ---")

df_surv <- df %>%
  filter(!is.na(time_surv), !is.na(event_surv), time_surv > 0)

## 7a. Overall KM
km_overall <- survfit(Surv(time_surv, event_surv) ~ 1, data = df_surv)

gsp_overall <- ggsurvplot(
  km_overall, data = df_surv,
  risk.table = TRUE, conf.int = TRUE,
  palette    = "#1976D2",
  xlab       = "Follow-up (years)",
  ylab       = "Overall Survival Probability",
  title      = "Overall Survival - Medullary Thyroid Carcinoma",
  ggtheme    = theme_classic(base_size = 12),
  risk.table.height = 0.25
)
open_png("Fig3a_KM_Overall.png")
print(gsp_overall)
dev.off()
message("[OK] Fig3a_KM_Overall.png")

## 7b. KM by disease type
km_dtype <- survfit(Surv(time_surv, event_surv) ~ disease_type_label,
                    data = df_surv)
lr_dtype  <- survdiff(Surv(time_surv, event_surv) ~ disease_type_label,
                      data = df_surv)
p_dtype   <- 1 - pchisq(lr_dtype$chisq, df = length(lr_dtype$n) - 1)

gsp_dtype <- ggsurvplot(
  km_dtype, data = df_surv,
  risk.table  = TRUE, conf.int = TRUE, pval = FALSE,
  palette     = c("#1976D2", "#D32F2F"),
  legend.labs = c("Sporadic", "Hereditary"),
  xlab        = "Follow-up (years)",
  ylab        = "Overall Survival Probability",
  title       = "Survival by Disease Type",
  ggtheme     = theme_classic(base_size = 12),
  risk.table.height = 0.25
)
gsp_dtype$plot <- gsp_dtype$plot +
  annotate("text", x = Inf, y = 0.08, hjust = 1.1, size = 4,
           label = paste0("Log-rank p = ", format.pval(p_dtype, digits = 3, eps = 0.001)))
open_png("Fig3b_KM_DiseaseType.png")
print(gsp_dtype)
dev.off()
message("[OK] Fig3b_KM_DiseaseType.png")

## 7c. KM by stage
km_stage <- survfit(Surv(time_surv, event_surv) ~ stage_label,
                    data = df_surv)
lr_stage  <- survdiff(Surv(time_surv, event_surv) ~ stage_label,
                      data = df_surv)
p_stage   <- 1 - pchisq(lr_stage$chisq, df = length(lr_stage$n) - 1)

gsp_stage <- ggsurvplot(
  km_stage, data = df_surv,
  risk.table  = TRUE, conf.int = FALSE, pval = FALSE,
  palette     = c("#388E3C", "#1976D2", "#FF9800", "#D32F2F"),
  xlab        = "Follow-up (years)",
  ylab        = "Overall Survival Probability",
  title       = "Survival by AJCC Stage",
  ggtheme     = theme_classic(base_size = 12),
  risk.table.height = 0.28
)
gsp_stage$plot <- gsp_stage$plot +
  annotate("text", x = Inf, y = 0.08, hjust = 1.1, size = 4,
           label = paste0("Log-rank p = ", format.pval(p_stage, digits = 3, eps = 0.001)))
open_png("Fig3c_KM_Stage.png", h = 9)
print(gsp_stage)
dev.off()
message("[OK] Fig3c_KM_Stage.png")

## 7d. KM by recurrence
km_recur <- survfit(Surv(time_surv, event_surv) ~ recurrence_num,
                    data = df_surv)
lr_recur  <- survdiff(Surv(time_surv, event_surv) ~ recurrence_num,
                      data = df_surv)
p_recur   <- 1 - pchisq(lr_recur$chisq, df = length(lr_recur$n) - 1)

gsp_recur <- ggsurvplot(
  km_recur, data = df_surv,
  risk.table  = TRUE, conf.int = FALSE, pval = FALSE,
  palette     = c("#388E3C", "#D32F2F"),
  legend.labs = c("No Recurrence", "Recurrence"),
  xlab        = "Follow-up (years)",
  ylab        = "Overall Survival Probability",
  title       = "Survival by Recurrence Status",
  ggtheme     = theme_classic(base_size = 12),
  risk.table.height = 0.25
)
gsp_recur$plot <- gsp_recur$plot +
  annotate("text", x = Inf, y = 0.08, hjust = 1.1, size = 4,
           label = paste0("Log-rank p = ", format.pval(p_recur, digits = 3, eps = 0.001)))
open_png("Fig3d_KM_Recurrence.png")
print(gsp_recur)
dev.off()
message("[OK] Fig3d_KM_Recurrence.png")

## 7e. Cox PH model
cox_vars <- c("age_at_diagnosis", "sex", "stage",
              "metastasis_at_diagnosis", "recurrence",
              "calcitonin_preop", "cea_preop",
              "tumor_size_mm", "disease_type", "ret_mutation")

df_cox <- df_surv %>%
  select(time_surv, event_surv, all_of(cox_vars)) %>%
  mutate(across(c(sex, stage, metastasis_at_diagnosis,
                  recurrence, disease_type, ret_mutation),
                as.numeric)) %>%
  na.omit()

cox_model <- coxph(
  Surv(time_surv, event_surv) ~
    age_at_diagnosis + sex + stage +
    metastasis_at_diagnosis + recurrence +
    log1p(calcitonin_preop) + log1p(cea_preop) +
    tumor_size_mm + disease_type + ret_mutation,
  data = df_cox
)
cox_sum <- summary(cox_model)
cox_ci  <- confint(cox_model)

cox_res <- data.frame(
  Variable    = rownames(cox_sum$coefficients),
  HR          = round(exp(cox_sum$coefficients[, "coef"]), 3),
  CI_Lower    = round(exp(cox_ci[, 1]), 3),
  CI_Upper    = round(exp(cox_ci[, 2]), 3),
  z           = round(cox_sum$coefficients[, "z"], 3),
  p_value     = ifelse(cox_sum$coefficients[, "Pr(>|z|)"] < 0.001,
                       "<0.001",
                       formatC(cox_sum$coefficients[, "Pr(>|z|)"],
                               digits = 3, format = "f")),
  stringsAsFactors = FALSE
)
cox_res[["HR (95% CI)"]] <- sprintf("%.2f (%.2f-%.2f)",
                                    cox_res$HR,
                                    cox_res$CI_Lower,
                                    cox_res$CI_Upper)
save_csv(cox_res, "Table2_Cox_PH_Results.csv")

## Forest plot (manual ggplot2 - robust to extreme HRs)
fp_df <- data.frame(
  Variable = rownames(cox_sum$coefficients),
  HR       = exp(cox_sum$coefficients[, "coef"]),
  Lower    = exp(cox_ci[, 1]),
  Upper    = exp(cox_ci[, 2]),
  pval     = cox_sum$coefficients[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
) %>%
  mutate(
    Lower   = pmax(Lower, 0.05),
    Upper   = pmin(Upper, 50),
    HR_plot = pmax(pmin(HR, 50), 0.05),
    Sig     = ifelse(pval < 0.05, "p < 0.05", "p >= 0.05"),
    Label   = recode(Variable,
      age_at_diagnosis        = "Age at diagnosis",
      sex                     = "Male sex",
      stage                   = "AJCC Stage",
      metastasis_at_diagnosis = "Metastasis at diagnosis",
      recurrence              = "Recurrence",
      `log1p(calcitonin_preop)` = "log(Calcitonin preop)",
      `log1p(cea_preop)`        = "log(CEA preop)",
      tumor_size_mm           = "Tumor size (mm)",
      disease_type            = "Hereditary disease",
      ret_mutation            = "RET mutation"),
    pval_lab = ifelse(pval < 0.001, "<0.001",
                      formatC(pval, digits = 3, format = "f"))
  ) %>%
  arrange(HR_plot)
fp_df$Label <- factor(fp_df$Label, levels = fp_df$Label)

p_forest <- ggplot(fp_df, aes(x = HR_plot, y = Label, colour = Sig)) +
  geom_vline(xintercept = 1, linetype = "dashed",
             colour = "grey50", linewidth = 0.7) +
  geom_errorbarh(aes(xmin = Lower, xmax = Upper),
                 height = 0.25, linewidth = 0.8) +
  geom_point(size = 3.5, shape = 18) +
  geom_text(aes(label = sprintf("HR %.2f (%.2f-%.2f) p=%s",
                                HR, Lower, Upper, pval_lab)),
            hjust = -0.1, size = 2.8, colour = "grey30") +
  scale_colour_manual(values = c("p < 0.05" = "#D32F2F",
                                 "p >= 0.05" = "#1976D2")) +
  scale_x_log10(limits = c(0.04, 80),
                breaks  = c(0.1, 0.25, 0.5, 1, 2, 5, 10, 25),
                labels  = c("0.1","0.25","0.5","1","2","5","10","25")) +
  labs(title    = "Cox PH - Overall Survival",
       subtitle = "Hazard Ratios with 95% CI (log scale)",
       x = "Hazard Ratio (log scale)", y = NULL, colour = NULL) +
  theme_classic(base_size = 12) +
  theme(legend.position    = "top",
        plot.title         = element_text(face = "bold"),
        axis.text.y        = element_text(size = 10),
        panel.grid.major.x = element_line(colour = "grey90",
                                          linewidth = 0.4))
save_fig(p_forest, "Fig4_Cox_ForestPlot.png")

# ============================================================
# 8. ML DATA PREPARATION
# ============================================================
message("\n--- [6/15] ML Data Preparation ---")

ml_vars <- c(
  "age_at_diagnosis", "sex", "disease_type", "ret_mutation",
  "tumor_size_mm", "stage", "multifocal", "lymph_node_invasion",
  "capsular_invasion", "soft_tissue_invasion",
  "metastasis_at_diagnosis", "calcitonin_preop", "cea_preop",
  "lymph_node_metastasis", "lung_metastasis_present",
  "bone_metastasis_present", "liver_metastasis_present"
)

df_ml <- df %>%
  select(recurrence, all_of(ml_vars)) %>%
  mutate(across(all_of(ml_vars) & where(is.factor), as.numeric)) %>%
  na.omit()

df_ml$recurrence <- factor(
  as.integer(as.character(df_ml$recurrence)),
  levels = c(0, 1), labels = c("Rec0", "Rec1")
)
df_ml <- df_ml %>% filter(!is.na(recurrence))

cat(sprintf("ML dataset: n=%d | Rec1=%d | Rec0=%d\n",
            nrow(df_ml),
            sum(df_ml$recurrence == "Rec1"),
            sum(df_ml$recurrence == "Rec0")))

# Stratified 70/30 split
yes_idx   <- which(df_ml$recurrence == "Rec1")
no_idx    <- which(df_ml$recurrence == "Rec0")
train_yes <- sample(yes_idx, size = max(1, round(0.70 * length(yes_idx))))
train_no  <- sample(no_idx,  size = max(1, round(0.70 * length(no_idx))))
train_df  <- df_ml[c(train_yes, train_no), ]
test_df   <- df_ml[-c(train_yes, train_no), ]

# ROSE balancing
train_bal <- tryCatch(
  ROSE(recurrence ~ ., data = train_df, seed = 42,
       N = 2 * nrow(train_df))$data,
  error = function(e) {
    message("ROSE failed - using upSample")
    upSample(x     = train_df[, -1],
             y     = train_df$recurrence,
             yname = "recurrence")
  }
)

cat(sprintf("Train (balanced): Rec1=%d | Rec0=%d\n",
            sum(train_bal$recurrence == "Rec1"),
            sum(train_bal$recurrence == "Rec0")))

# CV control (5-fold repeated x3)
ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter     = FALSE
)

# ============================================================
# 9. MODEL TRAINING WITH TIMING
# ============================================================
message("\n--- [7/15] Training 9 ML Models ---")

timing  <- list()
models  <- list()

timed_train <- function(name, expr_str) {
  t0 <- proc.time()
  m  <- eval(parse(text = expr_str))
  elapsed <- round((proc.time() - t0)["elapsed"], 2)
  timing[[name]] <<- elapsed
  models[[name]] <<- m
  message(sprintf("  [OK] %-14s %.1fs", name, elapsed))
}

# WHITE-BOX MODELS
timed_train("LASSO",
  'train(recurrence ~ ., data = train_bal, method = "glmnet",
   family = "binomial", metric = "ROC", trControl = ctrl,
   tuneLength = 10, preProcess = c("center","scale"))')

timed_train("CART",
  'train(recurrence ~ ., data = train_bal, method = "rpart",
   metric = "ROC", trControl = ctrl, tuneLength = 10,
   control = rpart.control(maxdepth = 5, minsplit = 10))')

timed_train("NaiveBayes",
  'train(recurrence ~ ., data = train_bal, method = "naive_bayes",
   metric = "ROC", trControl = ctrl, tuneLength = 5)')

timed_train("KNN",
  'train(recurrence ~ ., data = train_bal, method = "knn",
   metric = "ROC", trControl = ctrl,
   tuneGrid = data.frame(k = seq(1, 21, by = 2)),
   preProcess = c("center","scale"))')

# BLACK-BOX MODELS
timed_train("RandomForest",
  'train(recurrence ~ ., data = train_bal, method = "rf",
   metric = "ROC", trControl = ctrl, tuneLength = 5, ntree = 500)')

timed_train("SVM",
  'train(recurrence ~ ., data = train_bal, method = "svmRadial",
   metric = "ROC", trControl = ctrl, tuneLength = 5,
   preProcess = c("center","scale"))')

timed_train("NeuralNet",
  'train(recurrence ~ ., data = train_bal, method = "nnet",
   metric = "ROC", trControl = ctrl, tuneLength = 5,
   preProcess = c("center","scale"), trace = FALSE, MaxNWts = 500)')

timed_train("GBM",
  'train(recurrence ~ ., data = train_bal, method = "gbm",
   metric = "ROC", trControl = ctrl, tuneLength = 5, verbose = FALSE)')

# XGBoost (native xgb.train - v2.0 compatible)
xgb_feat  <- setdiff(names(train_bal), "recurrence")
xgb_tr    <- xgb.DMatrix(as.matrix(train_bal[, xgb_feat]),
                          label = as.numeric(train_bal$recurrence == "Rec1"))
xgb_te    <- xgb.DMatrix(as.matrix(test_df[, xgb_feat]),
                          label = as.numeric(test_df$recurrence == "Rec1"))
xgb_par   <- list(objective = "binary:logistic", eval_metric = "auc",
                  max_depth = 3, eta = 0.1, subsample = 0.8,
                  colsample_bytree = 0.8, nthread = 1)
t0_xgb    <- proc.time()
xgb_cv    <- xgb.cv(params = xgb_par, data = xgb_tr, nrounds = 200,
                    nfold = 5, early_stopping_rounds = 20,
                    verbose = 0, print_every_n = 9999)
best_nr   <- xgb_cv$best_iteration
if (is.null(best_nr) || is.na(best_nr) || best_nr < 1) best_nr <- 50
m_xgb     <- xgb.train(params = xgb_par, data = xgb_tr,
                        nrounds = best_nr, verbose = 0, print_every_n = 9999)
elapsed_xgb <- round((proc.time() - t0_xgb)["elapsed"], 2)
timing[["XGBoost"]] <- elapsed_xgb
models[["XGBoost"]] <- m_xgb
message(sprintf("  [OK] %-14s %.1fs (nrounds=%d)", "XGBoost", elapsed_xgb, best_nr))

# Timing dataframe
timing_df <- data.frame(
  Model   = names(timing),
  Seconds = as.numeric(unlist(timing)),
  Type    = ifelse(names(timing) %in% WHITEBOX, "WhiteBox", "BlackBox"),
  stringsAsFactors = FALSE
)

# ============================================================
# 10. MODEL EVALUATION
# ============================================================
message("\n--- [8/15] Model Evaluation ---")

eval_one <- function(nm, is_xgb = FALSE) {
  mdl <- models[[nm]]
  if (is_xgb) {
    prob <- predict(mdl, xgb_te)
    pred <- factor(ifelse(prob >= 0.5, "Rec1", "Rec0"),
                   levels = c("Rec0", "Rec1"))
  } else {
    pred <- predict(mdl, test_df)
    prob <- predict(mdl, test_df, type = "prob")[, "Rec1"]
  }
  cm  <- confusionMatrix(pred, test_df$recurrence, positive = "Rec1")
  roc <- pROC::roc(as.numeric(test_df$recurrence == "Rec1"),
                   as.numeric(prob), quiet = TRUE)
  data.frame(
    Model       = nm,
    Type        = ifelse(nm %in% WHITEBOX, "WhiteBox", "BlackBox"),
    AUC         = round(as.numeric(pROC::auc(roc)), 3),
    Accuracy    = round(cm$overall["Accuracy"], 3),
    Sensitivity = round(cm$byClass["Sensitivity"], 3),
    Specificity = round(cm$byClass["Specificity"], 3),
    PPV         = round(cm$byClass["Pos Pred Value"], 3),
    NPV         = round(cm$byClass["Neg Pred Value"], 3),
    F1          = round(cm$byClass["F1"], 3),
    row.names   = NULL
  )
}

all_nms <- c(WHITEBOX, setdiff(BLACKBOX, "XGBoost"), "XGBoost")
eval_list <- lapply(all_nms, function(nm) eval_one(nm, is_xgb = (nm == "XGBoost")))
eval_df   <- do.call(rbind, eval_list)
rownames(eval_df) <- NULL
eval_df   <- eval_df[order(eval_df$AUC, decreasing = TRUE), ]

# Add training time
eval_df <- left_join(eval_df, timing_df[, c("Model","Seconds")], by = "Model")

save_csv(eval_df, "Table3_ML_Performance.csv")
cat("\nModel Performance:\n")
print(eval_df[, c("Model","Type","AUC","Sensitivity","Specificity","F1","Seconds")])

# ============================================================
# 11. TRAINING TIME FIGURES
# ============================================================
message("\n--- [9/15] Training Time Figures ---")

## Fig 5A: Bar chart
timing_sorted <- timing_df %>% arrange(desc(Seconds)) %>%
  mutate(Model = factor(Model, levels = rev(Model)))

p_time_bar <- ggplot(timing_sorted,
                     aes(x = Model, y = Seconds, fill = Type)) +
  geom_bar(stat = "identity", alpha = 0.88, color = "white", width = 0.65) +
  geom_text(aes(label = paste0(Seconds, "s")),
            hjust = -0.1, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("WhiteBox" = WB_COL, "BlackBox" = BB_COL)) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title    = "A. Training Time by Model",
       subtitle = "Seed=42 | 5-fold repeated CV x3 | ROSE balancing",
       x = NULL, y = "Elapsed Time (seconds)", fill = "Model Type") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

## Fig 5B: Time vs AUC scatter
timing_auc <- left_join(timing_df, eval_df[, c("Model","AUC")], by = "Model")

p_time_auc <- ggplot(timing_auc,
                     aes(x = Seconds, y = AUC, color = Type, label = Model)) +
  geom_point(size = 5, alpha = 0.85) +
  geom_text(vjust = -0.9, size = 3.2, show.legend = FALSE) +
  geom_hline(yintercept = 0.7, lty = 2, color = "gray60") +
  scale_color_manual(values = c("WhiteBox" = WB_COL, "BlackBox" = BB_COL)) +
  labs(title = "B. Training Time vs. Test AUC",
       x = "Training Time (seconds)", y = "Test AUC",
       color = "Model Type") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")

fig5 <- plot_grid(p_time_bar, p_time_auc, ncol = 2, rel_widths = c(1.1, 0.9))
save_fig(fig5, "Fig5_Training_Times.png", w = 14, h = 7)

# ============================================================
# 12. WHITEBOX FIGURES
# ============================================================
message("\n--- [10/15] WhiteBox Figures ---")

## Fig 6A: LASSO coefficients
best_lambda <- models$LASSO$bestTune$lambda
coefs <- coef(models$LASSO$finalModel, s = best_lambda)
lasso_df <- data.frame(
  Variable    = rownames(coefs),
  Coefficient = as.numeric(coefs),
  OddsRatio   = exp(as.numeric(coefs))
) %>%
  filter(Variable != "(Intercept)", Coefficient != 0) %>%
  arrange(desc(abs(Coefficient))) %>%
  mutate(Direction = ifelse(Coefficient > 0, "Positive", "Negative"),
         Variable  = factor(Variable, levels = Variable[order(Coefficient)]))

save_csv(lasso_df, "Table4_LASSO_Coefficients.csv")

p_lasso_coef <- ggplot(lasso_df,
                       aes(x = Variable, y = Coefficient, fill = Direction)) +
  geom_bar(stat = "identity", alpha = 0.85) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("Positive" = "#D32F2F",
                               "Negative" = WB_COL)) +
  coord_flip() +
  labs(title    = "A. LASSO Coefficients (Log-Odds)",
       subtitle = paste0("lambda = ", round(best_lambda, 5),
                         " | alpha = 1 (L1)"),
       x = NULL, y = "Coefficient") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")

## Fig 6B: LASSO regularization path
open_png("Fig6B_LASSO_RegPath.png")
plot(models$LASSO$finalModel, xvar = "lambda", label = TRUE,
     main = "B. LASSO Regularization Path")
abline(v   = log(best_lambda), col = "red", lty = 2, lwd = 2)
legend("topright", legend = paste("Best lambda:", round(best_lambda, 5)),
       col = "red", lty = 2, bty = "n")
dev.off()
message("[OK] Fig6B_LASSO_RegPath.png")

## Fig 6C: CART Decision Tree
open_png("Fig6C_CART_DecisionTree.png", w = 14, h = 10)
rpart.plot(models$CART$finalModel, type = 4, extra = 104,
           roundint = FALSE, fallen.leaves = TRUE,
           main = "C. CART Decision Tree - MTC Recurrence",
           cex = 0.75, box.palette = "RdYlGn")
dev.off()
message("[OK] Fig6C_CART_DecisionTree.png")

## CART rules
cart_rules <- capture.output(
  rpart.rules(models$CART$finalModel, cover = TRUE)
)
writeLines(cart_rules, file.path(OUT, "Table5_CART_Rules.txt"))
message("[OK] Table5_CART_Rules.txt")

## Fig 6D: NB likelihood for top feature
nb_top <- "calcitonin_preop"
nb_lik_df <- train_bal %>%
  mutate(feat = as.numeric(.data[[nb_top]])) %>%
  filter(!is.na(feat))

p_nb <- ggplot(nb_lik_df, aes(x = feat, fill = recurrence)) +
  geom_density(alpha = 0.65, color = "white") +
  scale_fill_manual(values = c("Rec0" = WB_COL, "Rec1" = "#D32F2F"),
                    labels = c("No Recurrence", "Recurrence")) +
  scale_x_log10() +
  labs(title    = "D. Naive Bayes - P(Calcitonin | Class)",
       subtitle = "Feature likelihood density by recurrence class",
       x = "Calcitonin Preop (pg/mL, log scale)",
       y = "Density", fill = "Class") +
  theme_classic(base_size = 12)

## Fig 6E: KNN k tuning
knn_res <- models$KNN$results
best_k  <- models$KNN$bestTune$k
p_knn <- ggplot(knn_res, aes(x = k, y = ROC)) +
  geom_line(color = WB_COL, linewidth = 1.2) +
  geom_point(color = WB_COL, size = 3) +
  geom_vline(xintercept = best_k, lty = 2, color = "red", linewidth = 1) +
  annotate("text", x = best_k + 0.5, y = min(knn_res$ROC),
           label = paste("Best k =", best_k), hjust = 0, color = "red") +
  labs(title = "E. KNN - CV AUC vs k",
       x = "k (Neighbors)", y = "CV AUC") +
  theme_classic(base_size = 12)

## Combine WhiteBox figure
fig6_top <- plot_grid(p_lasso_coef, p_nb, ncol = 2, labels = c("A", "D"))
fig6_bot <- plot_grid(p_knn, ncol = 1, labels = "E")
fig6 <- plot_grid(fig6_top, fig6_bot, nrow = 2, rel_heights = c(1, 0.8))
save_fig(fig6, "Fig6_WhiteBox_Outputs.png", w = 14, h = 12)

# ============================================================
# 13. ROC CURVES
# ============================================================
message("\n--- [11/15] ROC Curves ---")

pal10 <- c("#1565C0","#388E3C","#F57C00","#7B1FA2",
           "#B71C1C","#00838F","#6A1B9A","#2E7D32","#37474F")

roc_list <- lapply(all_nms, function(nm) {
  prob <- if (nm == "XGBoost") predict(models$XGBoost, xgb_te)
          else predict(models[[nm]], test_df, type = "prob")[, "Rec1"]
  pROC::roc(as.numeric(test_df$recurrence == "Rec1"),
            as.numeric(prob), quiet = TRUE)
})
names(roc_list) <- all_nms
aucs <- sapply(roc_list, function(r) round(as.numeric(pROC::auc(r)), 3))

open_png("Fig7_ROC_Curves.png")
plot(roc_list[[1]], col = pal10[1], lwd = 2.5,
     main = "ROC Curves - 9 ML Models (Recurrence Prediction)",
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)",
     cex.main = 1.2, cex.lab = 1.1)
for (i in 2:length(roc_list))
  plot(roc_list[[i]], col = pal10[i], lwd = 2.5, add = TRUE)
legend("bottomright",
       legend = sprintf("%s  AUC=%.3f", names(aucs), aucs),
       col = pal10[seq_along(aucs)], lwd = 2.5, bty = "n", cex = 0.85)
abline(a = 0, b = 1, lty = 2, col = "gray60")
dev.off()
message("[OK] Fig7_ROC_Curves.png")

# ============================================================
# 14. BLACKBOX XAI FIGURES
# ============================================================
message("\n--- [12/15] BlackBox XAI Figures ---")

## Fig 8A: XGBoost SHAP summary
x_test_mat <- as.matrix(test_df[, xgb_feat])
shap_vals  <- predict(models$XGBoost, xgb.DMatrix(x_test_mat),
                      predcontrib = TRUE)
shap_vals  <- shap_vals[, colnames(shap_vals) != "BIAS"]

shap_long <- as.data.frame(shap_vals) %>%
  mutate(obs = row_number()) %>%
  pivot_longer(-obs, names_to = "Feature", values_to = "SHAP") %>%
  left_join(
    as.data.frame(x_test_mat) %>%
      mutate(obs = row_number()) %>%
      pivot_longer(-obs, names_to = "Feature", values_to = "Value"),
    by = c("obs", "Feature")
  ) %>%
  group_by(Feature) %>%
  mutate(MeanAbsSHAP = mean(abs(SHAP))) %>%
  ungroup() %>%
  mutate(Feature = reorder(Feature, MeanAbsSHAP))

p_shap_sum <- ggplot(shap_long, aes(x = Feature, y = SHAP, color = Value)) +
  geom_jitter(width = 0.2, alpha = 0.7, size = 2) +
  coord_flip() +
  scale_color_gradient(low = WB_COL, high = "#D32F2F",
                       name = "Feature\nValue") +
  geom_hline(yintercept = 0, lty = 2, color = "gray50") +
  labs(title    = "A. SHAP Summary - XGBoost",
       subtitle = "Each dot = one test patient",
       x = NULL, y = "SHAP Value") +
  theme_classic(base_size = 11)

## Fig 8B: Mean |SHAP| importance
shap_imp <- colMeans(abs(shap_vals)) %>%
  sort(decreasing = TRUE) %>%
  as.data.frame() %>%
  setNames("MeanAbsSHAP") %>%
  rownames_to_column("Feature") %>%
  head(12) %>%
  mutate(Feature = factor(Feature, levels = rev(Feature)))

save_csv(shap_imp, "Table6_SHAP_Importance.csv")

p_shap_imp <- ggplot(shap_imp,
                     aes(x = Feature, y = MeanAbsSHAP, fill = MeanAbsSHAP)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "#90CAF9", high = WB_COL) +
  labs(title    = "B. Mean |SHAP| - XGBoost Importance",
       subtitle = "Higher = greater average impact on prediction",
       x = NULL, y = "Mean |SHAP Value|") +
  theme_classic(base_size = 11) +
  theme(legend.position = "none")

## Fig 8C: XGBoost Gain importance
xgb_imp_df <- xgb.importance(feature_names = xgb_feat,
                              model = models$XGBoost) %>%
  as.data.frame() %>%
  head(12) %>%
  mutate(Feature = factor(Feature, levels = rev(Feature[order(Gain)])))

save_csv(xgb.importance(feature_names = xgb_feat, model = models$XGBoost),
         "Table7_XGBoost_Gain_Importance.csv")

p_xgb_imp <- ggplot(xgb_imp_df, aes(x = Feature, y = Gain, fill = Gain)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "#90CAF9", high = WB_COL) +
  labs(title = "C. XGBoost - Gain Importance",
       x = NULL, y = "Gain") +
  theme_classic(base_size = 11) +
  theme(legend.position = "none")

## Fig 8D: Random Forest importance
rf_imp_df <- varImp(models$RandomForest)$importance %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  head(12) %>%
  mutate(Variable = factor(Variable, levels = rev(Variable)))

save_csv(rf_imp_df, "Table8_RF_Gini_Importance.csv")

p_rf_imp <- ggplot(rf_imp_df, aes(x = Variable, y = Overall, fill = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "#A5D6A7", high = "#1B5E20") +
  labs(title = "D. Random Forest - Gini Importance",
       x = NULL, y = "Mean Decrease Gini") +
  theme_classic(base_size = 11) +
  theme(legend.position = "none")

## Fig 8E: GBM relative influence
gbm_imp_df <- summary(models$GBM$finalModel, plotit = FALSE) %>%
  as.data.frame() %>%
  head(12) %>%
  mutate(var = factor(var, levels = rev(var)))

save_csv(gbm_imp_df, "Table9_GBM_RelInfluence.csv")

p_gbm_imp <- ggplot(gbm_imp_df, aes(x = var, y = rel.inf, fill = rel.inf)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "#FFCC80", high = "#E65100") +
  labs(title = "E. GBM - Relative Influence",
       x = NULL, y = "Relative Influence (%)") +
  theme_classic(base_size = 11) +
  theme(legend.position = "none")

## Combine XAI panel
fig8_top <- plot_grid(p_shap_sum, p_shap_imp, ncol = 2, labels = c("A","B"))
fig8_bot <- plot_grid(p_xgb_imp, p_rf_imp, p_gbm_imp,
                      ncol = 3, labels = c("C","D","E"))
fig8 <- plot_grid(fig8_top, fig8_bot, nrow = 2, rel_heights = c(1.2, 1))
save_fig(fig8, "Fig8_BlackBox_XAI.png", w = 16, h = 13)

# ============================================================
# 15. CV COMPARISON & MODEL COMPARISON FIGURES
# ============================================================
message("\n--- [13/15] CV Comparison Figure ---")

## CV distribution (caret models only)
caret_models <- models[setdiff(names(models), "XGBoost")]
cv_res  <- resamples(caret_models)
cv_long <- cv_res$values %>%
  select(ends_with("ROC")) %>%
  pivot_longer(everything(), names_to = "Model", values_to = "AUC") %>%
  mutate(Model = gsub("~ROC$", "", Model),
         Type  = ifelse(Model %in% WHITEBOX, "WhiteBox", "BlackBox"))

p_cv <- ggplot(cv_long, aes(x = reorder(Model, AUC, median),
                             y = AUC, fill = Type)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21, width = 0.55) +
  geom_jitter(width = 0.12, alpha = 0.35, size = 1.2) +
  scale_fill_manual(values = c("WhiteBox" = WB_COL, "BlackBox" = BB_COL)) +
  labs(title    = "Cross-Validation AUC Distribution",
       subtitle = "5-fold repeated CV x3 | XGBoost via native xgb.cv (not shown)",
       x = NULL, y = "AUC (ROC)") +
  theme_classic(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "top")
save_fig(p_cv, "Fig9_CV_AUC_Distribution.png", w = 10, h = 7)

## AUC comparison bar
p_auc_bar <- eval_df %>%
  arrange(AUC) %>%
  mutate(Model = factor(Model, levels = Model)) %>%
  ggplot(aes(x = Model, y = AUC, fill = Type)) +
  geom_bar(stat = "identity", alpha = 0.88, color = "white", width = 0.65) +
  geom_text(aes(label = AUC), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("WhiteBox" = WB_COL, "BlackBox" = BB_COL)) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1.12),
                     expand  = expansion(mult = c(0, 0))) +
  labs(title = "Test AUC by Model",
       x = NULL, y = "AUC (Test Set)") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")
save_fig(p_auc_bar, "Fig10_AUC_Comparison.png", w = 9, h = 7)

# ============================================================
# 16. DALEX VARIABLE IMPORTANCE (RF + GBM)
# ============================================================
message("\n--- [14/15] DALEX Permutation Importance ---")

x_test_df <- test_df %>% select(-recurrence)
y_test     <- as.numeric(test_df$recurrence == "Rec1")

dalex_vi_list <- list()
for (nm in c("RandomForest", "GBM", "NeuralNet")) {
  tryCatch({
    exp  <- DALEX::explain(models[[nm]], data = x_test_df,
                           y = y_test, label = nm, verbose = FALSE)
    vi   <- model_parts(exp, type = "variable_importance",
                        B = 10, loss_function = loss_one_minus_auc)
    dalex_vi_list[[nm]] <- vi

    p_vi <- plot(vi) +
      labs(title = paste("DALEX Permutation Importance -", nm)) +
      theme_classic(base_size = 11)
    save_fig(p_vi,
             paste0("Fig11_DALEX_VI_", nm, ".png"),
             w = 9, h = 6)
  }, error = function(e) {
    message("  DALEX VI failed for ", nm, ": ", conditionMessage(e))
  })
}

# DALEX VI table (RF if available)
if ("RandomForest" %in% names(dalex_vi_list)) {
  vi_tbl <- dalex_vi_list$RandomForest %>%
    as.data.frame() %>%
    filter(permutation != 0) %>%
    group_by(variable) %>%
    summarise(mean_dropout_loss = round(mean(dropout_loss), 4),
              .groups = "drop") %>%
    arrange(desc(mean_dropout_loss))
  save_csv(vi_tbl, "Table10_DALEX_Permutation_VI.csv")
}

# ============================================================
# 17. CALCITONIN REDUCTION ANALYSIS
# ============================================================
message("\n--- [15/15] Calcitonin Reduction Analysis ---")

df_cal_ratio <- df %>%
  filter(!is.na(calcitonin_preop), !is.na(calcitonin_postop),
         calcitonin_preop > 0) %>%
  mutate(
    cal_reduction_pct = 100 * (calcitonin_preop - calcitonin_postop) /
                               calcitonin_preop,
    biochem_cure      = ifelse(calcitonin_postop <= 10,
                               "Biochemical Cure",
                               "Persistent Disease")
  )

p_cal <- ggplot(df_cal_ratio,
                aes(x = biochem_cure, y = cal_reduction_pct,
                    fill = biochem_cure)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 21) +
  geom_jitter(width = 0.15, alpha = 0.5) +
  scale_fill_manual(values = c("#388E3C", "#D32F2F")) +
  stat_compare_means(method = "wilcox.test",
                     label = "p.format", size = 4.5) +
  labs(title = "Calcitonin Reduction by Biochemical Outcome",
       x = "", y = "Calcitonin Reduction (%)") +
  theme_classic(base_size = 12) +
  theme(legend.position = "none")
save_fig(p_cal, "Fig12_Calcitonin_Reduction.png", w = 8, h = 6)

cal_summary <- df_cal_ratio %>%
  group_by(biochem_cure) %>%
  summarise(
    n          = n(),
    median_pct = round(median(cal_reduction_pct, na.rm = TRUE), 1),
    iqr_pct    = round(IQR(cal_reduction_pct, na.rm = TRUE), 1),
    .groups    = "drop"
  )
save_csv(cal_summary, "Table11_Calcitonin_Summary.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
cat("\n")
cat("========================================================\n")
cat("  ANALYSIS COMPLETE\n")
cat("========================================================\n")

outputs <- list.files(OUT, pattern = "^(Fig|Table).*\\.(png|csv|txt)$")
figs    <- outputs[grepl("\\.png$", outputs)]
tabs    <- outputs[grepl("\\.(csv|txt)$", outputs)]

cat(sprintf("\nFigures (%d, all 600 DPI):\n", length(figs)))
cat(paste0("  ", seq_along(figs), ". ", sort(figs), collapse = "\n"), "\n")

cat(sprintf("\nTables (%d):\n", length(tabs)))
cat(paste0("  ", seq_along(tabs), ". ", sort(tabs), collapse = "\n"), "\n")

cat("\nBest Model by AUC:\n")
best_row <- eval_df[1, ]
cat(sprintf("  %s (%s) | AUC=%.3f | Sensitivity=%.3f | F1=%.3f\n",
            best_row$Model, best_row$Type,
            best_row$AUC, best_row$Sensitivity, best_row$F1))

cat("\nTraining Times:\n")
tt <- timing_df %>% arrange(Seconds)
cat(paste0("  ", tt$Model, " (", tt$Type, "): ", tt$Seconds, "s",
           collapse = "\n"), "\n")
cat(sprintf("  Total: %.1fs\n", sum(tt$Seconds)))
cat("========================================================\n\n")

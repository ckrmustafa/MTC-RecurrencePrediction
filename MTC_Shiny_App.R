pkgs <- c(
  "shiny","shinydashboard","shinyWidgets","shinyjs","DT",
  "tidyverse","tableone","survival","survminer",
  "caret","randomForest","glmnet","pROC",
  "xgboost","e1071","ROSE","naivebayes","rpart","rpart.plot","nnet","gbm",
  "iml","DALEX","DALEXtra","ggplot2","ggpubr","ggrepel",
  "corrplot","cowplot","scales","viridis","plotly"
)
installed <- rownames(installed.packages())
to_install <- pkgs[!pkgs %in% installed]
if (length(to_install) > 0)
  install.packages(to_install, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(invisible(lapply(pkgs, library, character.only = TRUE)))

# ============================================================
# DATA PREP
# ============================================================
prep_data <- function() {
  df_raw <- read.csv("DATA_2.csv", stringsAsFactors = FALSE, na.strings = c("","NA"))
  df_raw <- df_raw[, !duplicated(names(df_raw))]
  num_force <- c("age_at_diagnosis","follow_up_year","tumor_size_mm","usg_size_mm",
                 "calcitonin_preop","calcitonin_postop","cea_preop","cea_postop",
                 "recurrence_time_months")
  factor_vars <- c("sex","disease_type","ret_mutation","pheochromocytoma","tumor_side",
    "multifocal","lymph_node_invasion","capsular_invasion","soft_tissue_invasion",
    "metastasis_at_diagnosis","mediastinal_metastasis","lung_metastasis_at_diagnosis",
    "metastasis_followup","lung_metastasis_followup","liver_metastasis","bone_metastasis",
    "stage","lymph_node_metastasis","bone_metastasis_present","lung_metastasis_present",
    "liver_metastasis_present","adrenal_metastasis","recurrence","tt_cnd_snd","tt_cnd",
    "total_thyroidectomy","completion_thyroidectomy","hemithyroidectomy","disease_status",
    "vital_status","alive_no_disease","alive_with_disease","alive_unknown_status",
    "death_mtc","tki_therapy","radiotherapy","lutetium_therapy","chemotherapy","mibg_therapy")
  df <- df_raw %>%
    mutate(across(all_of(num_force), ~as.numeric(as.character(.)))) %>%
    mutate(across(all_of(intersect(factor_vars, names(.))), as.factor)) %>%
    mutate(
      sex_label          = factor(sex, levels=c(0,1), labels=c("Female","Male")),
      disease_type_label = factor(disease_type, levels=c(0,1), labels=c("Sporadic","Hereditary")),
      stage_label        = factor(stage, levels=c(1,2,3,4), labels=c("I","II","III","IV")),
      outcome_death      = as.numeric(as.character(death_mtc)),
      recurrence_num     = as.numeric(as.character(recurrence)),
      time_surv          = as.numeric(as.character(follow_up_year)),
      event_surv         = as.numeric(as.character(death_mtc))
    )
  df
}

# ============================================================
# ML TRAINING WITH TIMING
# ============================================================
train_models <- function(df, params) {
  set.seed(params$seed)
  all_ml_vars <- c("age_at_diagnosis","sex","disease_type","ret_mutation",
    "tumor_size_mm","stage","multifocal","lymph_node_invasion","capsular_invasion",
    "soft_tissue_invasion","metastasis_at_diagnosis","calcitonin_preop","cea_preop",
    "lymph_node_metastasis","lung_metastasis_present","bone_metastasis_present",
    "liver_metastasis_present")
  ml_vars <- if (!is.null(params$features) && length(params$features) > 2) params$features else all_ml_vars

  df_ml <- df %>%
    select(recurrence, all_of(ml_vars)) %>%
    mutate(across(all_of(ml_vars) & where(is.factor), as.numeric)) %>%
    na.omit()
  df_ml$recurrence <- factor(as.integer(as.character(df_ml$recurrence)),
                             levels=c(0,1), labels=c("Rec0","Rec1"))
  df_ml <- df_ml %>% filter(!is.na(recurrence))

  yes_idx   <- which(df_ml$recurrence == "Rec1")
  no_idx    <- which(df_ml$recurrence == "Rec0")
  split_p   <- params$split_ratio / 100
  train_yes <- sample(yes_idx, size=max(1, round(split_p*length(yes_idx))))
  train_no  <- sample(no_idx,  size=max(1, round(split_p*length(no_idx))))
  train_bal_raw <- df_ml[c(train_yes, train_no), ]
  test_df   <- df_ml[-c(train_yes, train_no), ]

  train_bal <- if (params$balance_method == "ROSE") {
    tryCatch(ROSE(recurrence~., data=train_bal_raw, seed=params$seed,
                  N=2*nrow(train_bal_raw))$data,
             error=function(e) upSample(x=train_bal_raw[,-1],
                                        y=train_bal_raw$recurrence, yname="recurrence"))
  } else if (params$balance_method == "UpSample") {
    upSample(x=train_bal_raw[,-1], y=train_bal_raw$recurrence, yname="recurrence")
  } else { train_bal_raw }

  ctrl <- trainControl(
    method=params$cv_method, number=params$cv_folds,
    repeats=if (params$cv_method=="repeatedcv") params$cv_repeats else 1,
    classProbs=TRUE, summaryFunction=twoClassSummary,
    savePredictions="final", verboseIter=FALSE
  )

  timing  <- list()
  models  <- list()

  timed_train <- function(name, mdl_expr) {
    t0 <- proc.time()
    m  <- eval(mdl_expr, envir=parent.env(environment()))
    elapsed <- round((proc.time() - t0)["elapsed"], 2)
    timing[[name]]  <<- elapsed
    models[[name]]  <<- m
    message("[OK] ", name, " trained in ", elapsed, "s")
  }

  # WHITE-BOX
  timed_train("LASSO", quote(train(recurrence~., data=train_bal,
    method="glmnet", family="binomial", metric="ROC", trControl=ctrl,
    tuneLength=params$lasso_tune, preProcess=c("center","scale"))))

  timed_train("CART", quote(train(recurrence~., data=train_bal,
    method="rpart", metric="ROC", trControl=ctrl, tuneLength=params$cart_tune,
    control=rpart.control(maxdepth=params$cart_maxdepth, minsplit=params$cart_minsplit))))

  timed_train("NaiveBayes", quote(train(recurrence~., data=train_bal,
    method="naive_bayes", metric="ROC", trControl=ctrl, tuneLength=5)))

  timed_train("KNN", quote(train(recurrence~., data=train_bal,
    method="knn", metric="ROC", trControl=ctrl,
    tuneGrid=data.frame(k=seq(params$knn_k_min, params$knn_k_max, by=2)),
    preProcess=c("center","scale"))))

  # BLACK-BOX
  timed_train("RandomForest", quote(train(recurrence~., data=train_bal,
    method="rf", metric="ROC", trControl=ctrl, tuneLength=5, ntree=params$rf_ntree)))

  timed_train("SVM", quote(train(recurrence~., data=train_bal,
    method="svmRadial", metric="ROC", trControl=ctrl, tuneLength=params$svm_tune,
    preProcess=c("center","scale"))))

  timed_train("NeuralNet", quote(train(recurrence~., data=train_bal,
    method="nnet", metric="ROC", trControl=ctrl, tuneLength=5,
    preProcess=c("center","scale"), trace=FALSE, MaxNWts=params$nnet_maxwts,
    tuneGrid=expand.grid(size=params$nnet_size, decay=params$nnet_decay))))

  timed_train("GBM", quote(train(recurrence~., data=train_bal,
    method="gbm", metric="ROC", trControl=ctrl, tuneLength=params$gbm_tune,
    verbose=FALSE,
    tuneGrid=expand.grid(n.trees=params$gbm_trees, interaction.depth=params$gbm_depth,
                         shrinkage=params$gbm_shrinkage, n.minobsinnode=10))))

  # XGBoost native
  xgb_feat <- setdiff(names(train_bal), "recurrence")
  xgb_tr   <- xgb.DMatrix(as.matrix(train_bal[, xgb_feat]),
                           label=as.numeric(train_bal$recurrence=="Rec1"))
  xgb_te   <- xgb.DMatrix(as.matrix(test_df[, xgb_feat]),
                           label=as.numeric(test_df$recurrence=="Rec1"))
  xgb_par  <- list(objective="binary:logistic", eval_metric="auc",
                   max_depth=params$xgb_depth, eta=params$xgb_eta,
                   subsample=params$xgb_subsample,
                   colsample_bytree=params$xgb_colsample, nthread=1)
  t0 <- proc.time()
  xgb_cv <- xgb.cv(params=xgb_par, data=xgb_tr, nrounds=params$xgb_nrounds,
                   nfold=params$cv_folds, early_stopping_rounds=20,
                   verbose=0, print_every_n=9999)
  best_nr <- xgb_cv$best_iteration
  if (is.null(best_nr)||is.na(best_nr)||best_nr<1) best_nr <- 50
  m_xgb <- xgb.train(params=xgb_par, data=xgb_tr, nrounds=best_nr,
                      verbose=0, print_every_n=9999)
  elapsed_xgb <- round((proc.time()-t0)["elapsed"],2)
  timing[["XGBoost"]] <- elapsed_xgb
  models[["XGBoost"]] <- m_xgb
  message("[OK] XGBoost trained in ", elapsed_xgb, "s | nrounds=", best_nr)

  timing_df <- data.frame(
    Model   = names(timing),
    Seconds = as.numeric(unlist(timing)),
    Type    = ifelse(names(timing) %in% c("LASSO","CART","NaiveBayes","KNN"),
                     "WhiteBox","BlackBox"),
    stringsAsFactors = FALSE
  )

  list(models=models, timing=timing_df,
       whitebox=c("LASSO","CART","NaiveBayes","KNN"),
       blackbox=c("RandomForest","SVM","NeuralNet","GBM","XGBoost"),
       train_bal=train_bal, test_df=test_df,
       xgb_feat=xgb_feat, xgb_te=xgb_te,
       df_ml=df_ml, ml_vars=ml_vars, params=params)
}

# ============================================================
# EVALUATION HELPER
# ============================================================
eval_model <- function(mdl, test_df, model_name, is_xgb=FALSE, xgb_te=NULL) {
  if (is_xgb) {
    prob <- predict(mdl, xgb_te)
    pred <- factor(ifelse(prob>=0.5,"Rec1","Rec0"), levels=c("Rec0","Rec1"))
  } else {
    pred <- predict(mdl, test_df)
    prob <- predict(mdl, test_df, type="prob")[,"Rec1"]
  }
  cm  <- confusionMatrix(pred, test_df$recurrence, positive="Rec1")
  roc <- pROC::roc(as.numeric(test_df$recurrence=="Rec1"), as.numeric(prob), quiet=TRUE)
  data.frame(Model=model_name,
             AUC=round(as.numeric(pROC::auc(roc)),3),
             Accuracy=round(cm$overall["Accuracy"],3),
             Sensitivity=round(cm$byClass["Sensitivity"],3),
             Specificity=round(cm$byClass["Specificity"],3),
             PPV=round(cm$byClass["Pos Pred Value"],3),
             NPV=round(cm$byClass["Neg Pred Value"],3),
             F1=round(cm$byClass["F1"],3),
             row.names=NULL)
}

# ============================================================
# EXPORT HELPERS
# ============================================================
dl_plot_btn <- function(id, label="Download PNG (600 DPI)") {
  downloadButton(id, label, class="btn-sm btn-success",
                 style="margin:4px 0;")
}
dl_table_btn <- function(id, label="Download CSV") {
  downloadButton(id, label, class="btn-sm btn-info",
                 style="margin:4px 0;")
}

register_plot_dl <- function(output, id, plot_fn, fname="plot", w=10, h=8) {
  output[[id]] <- downloadHandler(
    filename=function() paste0(fname,"_",format(Sys.time(),"%Y%m%d_%H%M%S"),".png"),
    content=function(file) {
      png(file, width=w, height=h, units="in", res=600)
      p <- plot_fn(); if (!is.null(p)) print(p)
      dev.off()
    }
  )
}

register_table_dl <- function(output, id, data_fn, fname="table") {
  output[[id]] <- downloadHandler(
    filename=function() paste0(fname,"_",format(Sys.time(),"%Y%m%d_%H%M%S"),".csv"),
    content=function(file) write.csv(data_fn(), file, row.names=FALSE)
  )
}

export_bar <- function(...) {
  div(style="display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 8px 0;", ...)
}

# ============================================================
# UI
# ============================================================
ui <- dashboardPage(
  skin="blue",
  dashboardHeader(title=span(icon("dna")," MTC-ML v2.0"), titleWidth=270),
  dashboardSidebar(
    width=260,
    sidebarMenu(id="sidebar",
      menuItem("Overview",        tabName="overview",  icon=icon("chart-bar")),
      menuItem("Settings",        tabName="settings",  icon=icon("sliders")),
      menuItem("Data Explorer",   tabName="data",      icon=icon("table")),
      menuItem("Survival",        tabName="survival",  icon=icon("heart-pulse")),
      menuItem("Training Times",  tabName="timing",    icon=icon("stopwatch")),
      menuItem("WhiteBox Models", tabName="wb",        icon=icon("magnifying-glass"),
        menuSubItem("LASSO",         tabName="wb_lasso"),
        menuSubItem("Decision Tree", tabName="wb_cart"),
        menuSubItem("Naive Bayes",   tabName="wb_nb"),
        menuSubItem("KNN",           tabName="wb_knn")
      ),
      menuItem("BlackBox + XAI",  tabName="bb",        icon=icon("brain"),
        menuSubItem("Performance",   tabName="bb_perf"),
        menuSubItem("SHAP",          tabName="bb_shap"),
        menuSubItem("DALEX",         tabName="bb_dalex"),
        menuSubItem("Variable Imp.", tabName="bb_varimp")
      ),
      menuItem("ROC Comparison",  tabName="roc",       icon=icon("chart-line")),
      menuItem("Prediction Tool", tabName="predict",   icon=icon("stethoscope"))
    ),
    hr(),
    div(style="padding:10px;color:#aaa;font-size:11px;",
        "MTC ML Dashboard v2.0",br(),
        "WhiteBox + BlackBox + XAI",br(),
        "600 DPI Export | Settings Tab")
  ),
  dashboardBody(
    useShinyjs(),
    tags$head(tags$style(HTML("
      .content-wrapper{background:#f4f6f9;}
      .box{border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);}
      .skin-blue .main-header .logo{font-weight:700;font-size:13px;}
      .math-box{background:#1e1e2e;color:#cdd6f4;font-family:monospace;
                padding:14px;border-radius:8px;font-size:11px;
                white-space:pre-wrap;overflow-x:auto;line-height:1.5;}
      .wb-badge{display:inline-block;background:#1565C0;color:#fff;
                padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700;}
      .bb-badge{display:inline-block;background:#B71C1C;color:#fff;
                padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700;}
      .ph{background:#37474F;color:#fff;padding:5px 12px;border-radius:5px;
          margin:10px 0 4px 0;font-weight:700;font-size:12px;}
    "))),
    tabItems(

      # OVERVIEW ---------------------------------------------------
      tabItem("overview",
        fluidRow(
          valueBoxOutput("vb_n",     width=3),
          valueBoxOutput("vb_s4",    width=3),
          valueBoxOutput("vb_rec",   width=3),
          valueBoxOutput("vb_fup",   width=3)
        ),
        fluidRow(
          box(title="Model Map", width=12, solidHeader=TRUE, status="primary",
            fluidRow(
              column(6,
                h4(span(class="wb-badge","WHITE-BOX")," Interpretable"),
                tags$ul(
                  tags$li(strong("LASSO")," - Coefficients, odds ratios, regularization path"),
                  tags$li(strong("CART")," - Visual decision tree, split rules"),
                  tags$li(strong("Naive Bayes")," - Prior/likelihood tables"),
                  tags$li(strong("KNN")," - k tuning, PCA boundary")
                )
              ),
              column(6,
                h4(span(class="bb-badge","BLACK-BOX")," Complex + XAI"),
                tags$ul(
                  tags$li(strong("Random Forest")," - SHAP + Gini importance"),
                  tags$li(strong("SVM")," - DALEX breakdown"),
                  tags$li(strong("Neural Net")," - Permutation importance"),
                  tags$li(strong("GBM")," - Relative influence + PDP"),
                  tags$li(strong("XGBoost")," - Native SHAP values")
                )
              )
            )
          )
        ),
        fluidRow(
          box(title="Train Models", width=5, solidHeader=TRUE, status="warning",
            p("Set parameters in",strong("Settings"),"tab, then click Train."),
            actionButton("btn_train","Train All 9 Models",
                         icon=icon("play"), class="btn-warning btn-lg"),
            br(),br(),
            verbatimTextOutput("train_status")
          ),
          box(title="Performance Summary", width=7, solidHeader=TRUE, status="info",
            export_bar(dl_table_btn("dl_sum_csv","Download Summary CSV")),
            DT::dataTableOutput("tbl_summary")
          )
        )
      ),

      # SETTINGS ---------------------------------------------------
      tabItem("settings",
        fluidRow(
          box(title="Global & CV Settings", width=4, solidHeader=TRUE, status="primary",
            div(class="ph","Random Seed"),
            numericInput("s_seed","Seed:", 42, 1, 99999),
            div(class="ph","Train/Test Split"),
            sliderInput("s_split","Training Set (%):", 50, 90, 70, step=5),
            div(class="ph","Cross-Validation"),
            selectInput("s_cv","CV Method:",
              c("repeatedcv","cv","LOOCV","boot"), selected="repeatedcv"),
            numericInput("s_folds","Folds (k):", 5, 2, 20),
            numericInput("s_reps","Repeats:", 3, 1, 10),
            div(class="ph","Class Balancing"),
            selectInput("s_bal","Method:", c("ROSE","UpSample","None")),
            div(class="ph","Feature Selection"),
            checkboxGroupInput("s_feats","Include features:",
              choices=c("age_at_diagnosis","sex","disease_type","ret_mutation",
                "tumor_size_mm","stage","multifocal","lymph_node_invasion",
                "capsular_invasion","soft_tissue_invasion","metastasis_at_diagnosis",
                "calcitonin_preop","cea_preop","lymph_node_metastasis",
                "lung_metastasis_present","bone_metastasis_present","liver_metastasis_present"),
              selected=c("age_at_diagnosis","sex","disease_type","ret_mutation",
                "tumor_size_mm","stage","multifocal","lymph_node_invasion",
                "capsular_invasion","soft_tissue_invasion","metastasis_at_diagnosis",
                "calcitonin_preop","cea_preop","lymph_node_metastasis",
                "lung_metastasis_present","bone_metastasis_present","liver_metastasis_present"))
          ),
          box(title="WhiteBox Parameters", width=4, solidHeader=TRUE, status="info",
            div(class="ph",span(class="wb-badge","LASSO")),
            numericInput("s_la_alpha","Alpha (0=Ridge, 1=LASSO):", 1, 0, 1, step=0.1),
            numericInput("s_la_tune","Tune Length:", 10, 5, 30),
            hr(),
            div(class="ph",span(class="wb-badge","CART")),
            numericInput("s_cart_depth","Max Depth:", 5, 1, 15),
            numericInput("s_cart_split","Min Split:", 10, 2, 50),
            numericInput("s_cart_tune","Tune Length:", 10, 5, 20),
            hr(),
            div(class="ph",span(class="wb-badge","KNN")),
            numericInput("s_knn_min","k Min:", 1, 1, 20, step=2),
            numericInput("s_knn_max","k Max:", 21, 5, 51, step=2)
          ),
          box(title="BlackBox Parameters", width=4, solidHeader=TRUE, status="danger",
            div(class="ph",span(class="bb-badge","Random Forest")),
            numericInput("s_rf_trees","n Trees:", 500, 100, 2000, step=100),
            hr(),
            div(class="ph",span(class="bb-badge","SVM")),
            numericInput("s_svm_tune","Tune Length:", 5, 3, 15),
            hr(),
            div(class="ph",span(class="bb-badge","Neural Net")),
            numericInput("s_nn_maxwts","Max Weights:", 500, 100, 2000, step=100),
            numericInput("s_nn_size","Hidden Units:", 5, 1, 30),
            numericInput("s_nn_decay","Weight Decay:", 0.01, 0, 0.5, step=0.01),
            hr(),
            div(class="ph",span(class="bb-badge","GBM")),
            numericInput("s_gbm_trees","n.trees:", 100, 50, 500, step=50),
            numericInput("s_gbm_depth","Interaction Depth:", 3, 1, 8),
            numericInput("s_gbm_shrink","Shrinkage:", 0.1, 0.01, 0.3, step=0.01),
            numericInput("s_gbm_tune","Tune Length:", 5, 3, 10),
            hr(),
            div(class="ph",span(class="bb-badge","XGBoost")),
            numericInput("s_xgb_depth","Max Depth:", 3, 1, 10),
            numericInput("s_xgb_eta","Learning Rate:", 0.1, 0.01, 0.5, step=0.01),
            numericInput("s_xgb_sub","Subsample:", 0.8, 0.5, 1.0, step=0.05),
            numericInput("s_xgb_col","ColSample:", 0.8, 0.5, 1.0, step=0.05),
            numericInput("s_xgb_nr","Max nrounds:", 200, 50, 1000, step=50)
          )
        )
      ),

      # DATA -------------------------------------------------------
      tabItem("data",
        fluidRow(
          box(title="Dataset", width=12, solidHeader=TRUE, status="primary",
            export_bar(dl_table_btn("dl_data_csv","Download Data CSV")),
            DT::dataTableOutput("tbl_data"))
        ),
        fluidRow(
          box(title="Distribution", width=6, solidHeader=TRUE,
            selectInput("dist_var","Variable:",
              c("age_at_diagnosis","tumor_size_mm","calcitonin_preop",
                "cea_preop","follow_up_year")),
            export_bar(dl_plot_btn("dl_dist_png")),
            plotOutput("plt_dist", height=260)),
          box(title="Correlation Heatmap", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_corr_png")),
            plotOutput("plt_corr", height=260))
        )
      ),

      # SURVIVAL ---------------------------------------------------
      tabItem("survival",
        fluidRow(
          box(title="Kaplan-Meier", width=8, solidHeader=TRUE, status="primary",
            selectInput("km_grp","Stratify by:",
              c("Overall"="overall","Disease Type"="disease_type_label",
                "Stage"="stage_label","Recurrence"="recurrence_num")),
            export_bar(dl_plot_btn("dl_km_png")),
            plotOutput("plt_km", height=400)),
          box(title="Cox PH", width=4, solidHeader=TRUE,
            export_bar(dl_table_btn("dl_cox_csv")),
            verbatimTextOutput("cox_out"))
        )
      ),

      # TRAINING TIMES --------------------------------------------
      tabItem("timing",
        fluidRow(
          box(title="Training Time - Bar Chart", width=8, solidHeader=TRUE, status="primary",
            p("Wall-clock elapsed seconds per model, measured during training."),
            export_bar(
              dl_plot_btn("dl_time_bar_png","Bar Chart PNG (600 DPI)"),
              dl_plot_btn("dl_time_dot_png","Dot Plot PNG (600 DPI)"),
              dl_table_btn("dl_time_csv","Download CSV")
            ),
            plotOutput("plt_time_bar", height=360)
          ),
          box(title="Time vs AUC Trade-off", width=4, solidHeader=TRUE, status="info",
            export_bar(dl_plot_btn("dl_time_auc_png","Download PNG (600 DPI)")),
            plotOutput("plt_time_auc", height=360)
          )
        ),
        fluidRow(
          box(title="Timing Summary Table", width=12, solidHeader=TRUE,
            DT::dataTableOutput("tbl_timing"))
        )
      ),

      # WB: LASSO -------------------------------------------------
      tabItem("wb_lasso",
        fluidRow(box(width=12, solidHeader=TRUE, status="primary",
          title=span(span(class="wb-badge","WB")," LASSO Logistic Regression"),
          p("L1 penalty: ||y-Xb||^2 + lambda*||b||_1. Zero coef = variable excluded."))),
        fluidRow(
          box(title="Coefficients", width=6, solidHeader=TRUE, status="info",
            export_bar(dl_table_btn("dl_lasso_coef_csv")),
            DT::dataTableOutput("tbl_lasso_coef")),
          box(title="Coefficient Plot", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_lasso_coef_png")),
            plotOutput("plt_lasso_coef", height=300))
        ),
        fluidRow(
          box(title="Regularization Path", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_lasso_path_png")),
            plotOutput("plt_lasso_path", height=260)),
          box(title="Model Equation", width=6, solidHeader=TRUE,
            div(class="math-box", textOutput("txt_lasso_eq")))
        )
      ),

      # WB: CART --------------------------------------------------
      tabItem("wb_cart",
        fluidRow(box(width=12, solidHeader=TRUE, status="primary",
          title=span(span(class="wb-badge","WB")," CART Decision Tree"),
          p("Gini impurity splits. Each node = explicit rule."))),
        fluidRow(
          box(title="Tree Visualization", width=8, solidHeader=TRUE, status="info",
            export_bar(dl_plot_btn("dl_cart_tree_png")),
            plotOutput("plt_cart_tree", height=460)),
          box(title="Rules", width=4, solidHeader=TRUE,
            verbatimTextOutput("txt_cart_rules"))
        ),
        fluidRow(
          box(title="Variable Importance", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_cart_vi_png"),
                       dl_table_btn("dl_cart_vi_csv")),
            plotOutput("plt_cart_vi", height=260)),
          box(title="Confusion Matrix", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_cart_cm_png")),
            plotOutput("plt_cart_cm", height=260))
        )
      ),

      # WB: NB ----------------------------------------------------
      tabItem("wb_nb",
        fluidRow(box(width=12, solidHeader=TRUE, status="primary",
          title=span(span(class="wb-badge","WB")," Naive Bayes"),
          p("P(Y|X) ~ P(Y) x prod P(Xi|Y). Assumes feature independence."))),
        fluidRow(
          box(title="Priors", width=4, solidHeader=TRUE,
            verbatimTextOutput("txt_nb_prior")),
          box(title="Likelihood P(Feature|Class)", width=8, solidHeader=TRUE,
            selectInput("nb_feat","Feature:",
              c("age_at_diagnosis","tumor_size_mm","calcitonin_preop",
                "stage","lymph_node_invasion")),
            export_bar(dl_plot_btn("dl_nb_lik_png")),
            plotOutput("plt_nb_lik", height=270))
        )
      ),

      # WB: KNN ---------------------------------------------------
      tabItem("wb_knn",
        fluidRow(box(width=12, solidHeader=TRUE, status="primary",
          title=span(span(class="wb-badge","WB")," K-Nearest Neighbors"),
          p("Majority vote of k nearest samples. Distance: Euclidean."))),
        fluidRow(
          box(title="Optimal k", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_knn_k_png")),
            plotOutput("plt_knn_k", height=270)),
          box(title="PCA Projection", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_knn_pca_png")),
            plotOutput("plt_knn_pca", height=270))
        )
      ),

      # BB: PERFORMANCE -------------------------------------------
      tabItem("bb_perf",
        fluidRow(box(width=12, solidHeader=TRUE, status="danger",
          title=span(span(class="bb-badge","BB")," Performance Comparison"),
          export_bar(dl_table_btn("dl_perf_csv")),
          DT::dataTableOutput("tbl_perf"))),
        fluidRow(
          box(title="AUC Bars", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_auc_bar_png")),
            plotOutput("plt_auc_bar", height=320)),
          box(title="CV AUC Distribution", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_cv_box_png")),
            plotOutput("plt_cv_box", height=320))
        )
      ),

      # BB: SHAP --------------------------------------------------
      tabItem("bb_shap",
        fluidRow(box(width=12, solidHeader=TRUE, status="danger",
          title=span(span(class="bb-badge","XAI")," SHAP Values - XGBoost"),
          p("SHapley Additive exPlanations: cooperative game theory feature attribution."))),
        fluidRow(
          box(title="SHAP Beeswarm Summary", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_shap_sum_png")),
            plotOutput("plt_shap_sum", height=360)),
          box(title="Mean |SHAP| Importance", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_shap_imp_png"),
                       dl_table_btn("dl_shap_imp_csv")),
            plotOutput("plt_shap_imp", height=360))
        ),
        fluidRow(
          box(title="Waterfall - Individual Patient", width=12, solidHeader=TRUE,
            sliderInput("shap_obs","Patient (test set):", 1, 10, 1, step=1),
            export_bar(dl_plot_btn("dl_shap_wf_png")),
            plotOutput("plt_shap_wf", height=300))
        )
      ),

      # BB: DALEX -------------------------------------------------
      tabItem("bb_dalex",
        fluidRow(box(width=12, solidHeader=TRUE, status="danger",
          title=span(span(class="bb-badge","XAI")," DALEX Model-Agnostic Explanations"))),
        fluidRow(
          box(title="Settings", width=3, solidHeader=TRUE,
            selectInput("dalex_mdl","Model:",
              c("RandomForest","GBM","NeuralNet","XGBoost")),
            sliderInput("dalex_obs","Patient:", 1, 10, 1),
            actionButton("btn_dalex","Run Break-Down", class="btn-danger btn-block"),
            br(),
            export_bar(dl_plot_btn("dl_dalex_bd_png"))),
          box(title="Break-Down Plot", width=9, solidHeader=TRUE,
            plotOutput("plt_dalex_bd", height=360))
        ),
        fluidRow(
          box(title="Permutation Variable Importance", width=12, solidHeader=TRUE,
            fluidRow(
              column(3,
                selectInput("dalex_vi_mdl","Model:",
                  c("RandomForest","GBM","NeuralNet")),
                actionButton("btn_dalex_vi","Compute", class="btn-danger"),
                br(),br(),
                export_bar(dl_plot_btn("dl_dalex_vi_png"))),
              column(9, plotOutput("plt_dalex_vi", height=300))
            ))
        )
      ),

      # BB: VAR IMP -----------------------------------------------
      tabItem("bb_varimp",
        fluidRow(
          box(title="XGBoost - Gain", width=6, solidHeader=TRUE, status="danger",
            export_bar(dl_plot_btn("dl_vi_xgb_png"),dl_table_btn("dl_vi_xgb_csv")),
            plotOutput("plt_vi_xgb", height=300)),
          box(title="Random Forest - Gini", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_vi_rf_png"),dl_table_btn("dl_vi_rf_csv")),
            plotOutput("plt_vi_rf", height=300))
        ),
        fluidRow(
          box(title="GBM - Relative Influence", width=6, solidHeader=TRUE,
            export_bar(dl_plot_btn("dl_vi_gbm_png"),dl_table_btn("dl_vi_gbm_csv")),
            plotOutput("plt_vi_gbm", height=300)),
          box(title="Partial Dependence", width=6, solidHeader=TRUE,
            selectInput("pdp_feat","Feature:",
              c("calcitonin_preop","tumor_size_mm","age_at_diagnosis","stage","cea_preop")),
            selectInput("pdp_mdl","Model:", c("RandomForest","GBM")),
            actionButton("btn_pdp","Compute PDP", class="btn-danger"),
            export_bar(dl_plot_btn("dl_pdp_png")),
            plotOutput("plt_pdp", height=240))
        )
      ),

      # ROC -------------------------------------------------------
      tabItem("roc",
        fluidRow(
          box(title="ROC - All 9 Models", width=8, solidHeader=TRUE, status="primary",
            export_bar(dl_plot_btn("dl_roc_png","Download ROC PNG (600 DPI)")),
            plotOutput("plt_roc", height=460)),
          box(title="AUC Table", width=4, solidHeader=TRUE,
            export_bar(dl_table_btn("dl_auc_csv")),
            DT::dataTableOutput("tbl_auc"))
        )
      ),

      # PREDICT ---------------------------------------------------
      tabItem("predict",
        fluidRow(
          box(title="Patient Input", width=4, solidHeader=TRUE, status="warning",
            numericInput("pi_age","Age:", 45, 10, 90),
            selectInput("pi_sex","Sex:", c("Female"=0,"Male"=1)),
            selectInput("pi_dtype","Disease Type:", c("Sporadic"=0,"Hereditary"=1)),
            numericInput("pi_tumor","Tumor Size (mm):", 25, 1, 200),
            selectInput("pi_stage","Stage:", c("I"=1,"II"=2,"III"=3,"IV"=4)),
            selectInput("pi_lni","LN Invasion:", c("No"=0,"Yes"=1)),
            numericInput("pi_cal","Calcitonin Preop:", 1000, 0, 100000),
            numericInput("pi_cea","CEA Preop:", 200, 0, 10000),
            selectInput("pi_meta","Metastasis at Dx:", c("No"=0,"Yes"=1)),
            actionButton("btn_pred","Predict",
                         class="btn-warning btn-lg", icon=icon("calculator"))
          ),
          box(title="Results", width=8, solidHeader=TRUE, status="info",
            fluidRow(
              column(6,
                export_bar(dl_plot_btn("dl_pred_bar_png")),
                plotOutput("plt_pred_bar", height=300)),
              column(6,
                uiOutput("ui_consensus"),
                br(),
                export_bar(dl_plot_btn("dl_pred_exp_png")),
                plotOutput("plt_pred_exp", height=220))
            )
          )
        )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================
server <- function(input, output, session) {

  `%||%` <- function(a,b) if (!is.null(a) && length(a)>0 && !is.na(a[1])) a else b

  df <- reactive({ prep_data() })

  get_params <- reactive({
    list(
      seed=input$s_seed%||%42, split_ratio=input$s_split%||%70,
      cv_method=input$s_cv%||%"repeatedcv", cv_folds=input$s_folds%||%5,
      cv_repeats=input$s_reps%||%3, balance_method=input$s_bal%||%"ROSE",
      features=input$s_feats,
      lasso_alpha=input$s_la_alpha%||%1, lasso_tune=input$s_la_tune%||%10,
      cart_maxdepth=input$s_cart_depth%||%5, cart_minsplit=input$s_cart_split%||%10,
      cart_tune=input$s_cart_tune%||%10,
      knn_k_min=input$s_knn_min%||%1, knn_k_max=input$s_knn_max%||%21,
      rf_ntree=input$s_rf_trees%||%500, svm_tune=input$s_svm_tune%||%5,
      nnet_maxwts=input$s_nn_maxwts%||%500, nnet_size=input$s_nn_size%||%5,
      nnet_decay=input$s_nn_decay%||%0.01,
      gbm_trees=input$s_gbm_trees%||%100, gbm_depth=input$s_gbm_depth%||%3,
      gbm_shrinkage=input$s_gbm_shrink%||%0.1, gbm_tune=input$s_gbm_tune%||%5,
      xgb_depth=input$s_xgb_depth%||%3, xgb_eta=input$s_xgb_eta%||%0.1,
      xgb_subsample=input$s_xgb_sub%||%0.8, xgb_colsample=input$s_xgb_col%||%0.8,
      xgb_nrounds=input$s_xgb_nr%||%200
    )
  })

  ml <- eventReactive(input$btn_train, {
    withProgress(message="Training 9 models...", value=0, {
      incProgress(0.05, detail="Preparing data")
      r <- train_models(df(), get_params())
      incProgress(0.95, detail="Done")
      r
    })
  })

  # VALUE BOXES
  output$vb_n  <- renderValueBox(valueBox(nrow(df()),"Patients",icon=icon("users"),color="blue"))
  output$vb_s4 <- renderValueBox({
    n <- sum(df()$stage==4,na.rm=TRUE)
    valueBox(sprintf("%d (%.0f%%)",n,100*n/nrow(df())),"Stage IV",icon=icon("triangle-exclamation"),color="red")})
  output$vb_rec <- renderValueBox({
    n <- sum(df()$recurrence==1,na.rm=TRUE)
    valueBox(sprintf("%d (%.0f%%)",n,100*n/nrow(df())),"Recurrence",icon=icon("rotate"),color="orange")})
  output$vb_fup <- renderValueBox(
    valueBox(paste0(round(median(df()$follow_up_year,na.rm=TRUE),1)," yr"),
             "Median Follow-up",icon=icon("clock"),color="green"))

  output$train_status <- renderText({
    req(ml())
    m <- ml(); p <- m$params
    paste0("Trained: ",length(m$models)," models\n",
           "Seed: ",p$seed," | CV: ",p$cv_method," (k=",p$cv_folds,")\n",
           "Balance: ",p$balance_method," | Features: ",length(m$ml_vars),"\n",
           "Total time: ",sum(m$timing$Seconds),"s")
  })

  # ALL RESULTS reactive
  all_res <- reactive({
    req(ml()); m <- ml()
    all_nms <- c(m$whitebox, setdiff(m$blackbox,"XGBoost"),"XGBoost")
    rows <- lapply(all_nms, function(nm)
      eval_model(m$models[[nm]], m$test_df, nm,
                 is_xgb=(nm=="XGBoost"), xgb_te=m$xgb_te))
    res <- do.call(rbind,rows)
    res$Type <- ifelse(res$Model %in% m$whitebox,"WhiteBox","BlackBox")
    res[order(res$AUC,decreasing=TRUE),]
  })

  # SUMMARY TABLE
  fmt_dt <- function(df) {
    DT::datatable(df, rownames=FALSE, options=list(dom="t",pageLength=15)) %>%
      DT::formatStyle("Type",
        backgroundColor=DT::styleEqual(c("WhiteBox","BlackBox"),c("#E3F2FD","#FFEBEE")))
  }
  output$tbl_summary <- DT::renderDataTable({ req(all_res()); fmt_dt(all_res()) })
  register_table_dl(output,"dl_sum_csv", all_res, "performance_summary")

  # DATA
  data_display <- reactive({
    df() %>% select(age_at_diagnosis,sex_label,disease_type_label,
                    stage_label,tumor_size_mm,calcitonin_preop,
                    cea_preop,recurrence,follow_up_year)
  })
  output$tbl_data <- DT::renderDataTable({
    DT::datatable(data_display(), rownames=FALSE,
                  options=list(pageLength=15,scrollX=TRUE))
  })
  register_table_dl(output,"dl_data_csv", data_display, "patient_data")

  # Distribution
  dist_p <- reactive({
    ggplot(df(), aes_string(x=input$dist_var))+
      geom_histogram(fill="#1976D2",color="white",bins=20,alpha=0.8)+
      labs(title=paste("Distribution:",input$dist_var),x=input$dist_var,y="Count")+
      theme_classic(base_size=12)
  })
  output$plt_dist <- renderPlot({ dist_p() })
  register_plot_dl(output,"dl_dist_png", dist_p, "distribution")

  # Correlation
  corr_fn <- function() {
    nv <- c("age_at_diagnosis","tumor_size_mm","calcitonin_preop","cea_preop","follow_up_year")
    cm <- cor(df() %>% mutate(across(all_of(nv),~as.numeric(as.character(.)))) %>% select(all_of(nv)),
              use="pairwise.complete.obs",method="spearman")
    corrplot(cm,method="color",type="upper",tl.col="black",tl.cex=0.8,
             addCoef.col="black",number.cex=0.7,
             col=colorRampPalette(c("#D32F2F","white","#1976D2"))(200))
  }
  output$plt_corr <- renderPlot({ corr_fn() })
  output$dl_corr_png <- downloadHandler(
    filename=function() paste0("correlation_",format(Sys.time(),"%Y%m%d_%H%M%S"),".png"),
    content=function(file) { png(file,10,8,units="in",res=600); corr_fn(); dev.off() }
  )

  # SURVIVAL
  km_p <- reactive({
    d     <- df() %>% filter(!is.na(time_surv), !is.na(event_surv), time_surv > 0)
    grp   <- isolate(input$km_grp)          # isolate prevents reactive dependency issue
    is_overall <- (grp == "overall")

    if (is_overall) {
      fit <- survfit(Surv(time_surv, event_surv) ~ 1, data = d)
    } else {
      fml <- as.formula(paste("Surv(time_surv, event_surv) ~", grp))
      fit <- survfit(fml, data = d)
    }

    # Compute log-rank p-value manually (avoids surv_pvalue scoping issues)
    pval_label <- NULL
    if (!is_overall) {
      fml_lr <- as.formula(paste("Surv(time_surv, event_surv) ~", grp))
      lr_test <- survdiff(fml_lr, data = d)
      p_val   <- 1 - pchisq(lr_test$chisq, df = length(lr_test$n) - 1)
      pval_label <- paste0("Log-rank p = ", format.pval(p_val, digits = 3, eps = 0.001))
    }

    gsp <- ggsurvplot(
      fit, data = d,
      risk.table       = TRUE,
      conf.int         = TRUE,
      pval             = FALSE,          # disabled to avoid internal scoping error
      xlab             = "Follow-up (years)",
      ylab             = "Survival Probability",
      ggtheme          = theme_classic(base_size = 12),
      risk.table.height = 0.25
    )

    # Add p-value annotation manually if applicable
    if (!is.null(pval_label)) {
      gsp$plot <- gsp$plot +
        annotate("text", x = Inf, y = 0.05, hjust = 1.1,
                 label = pval_label, size = 4, color = "black",
                 fontface = "italic")
    }

    gsp$plot
  })
  output$plt_km <- renderPlot({ km_p() })
  register_plot_dl(output,"dl_km_png", km_p, "kaplan_meier")

  cox_res_df <- reactive({
    d <- df() %>% filter(!is.na(time_surv),!is.na(event_surv),time_surv>0) %>%
      mutate(stage_n=as.numeric(as.character(stage)),
             meta_n=as.numeric(as.character(metastasis_at_diagnosis)),
             recur_n=as.numeric(as.character(recurrence)))
    cox <- coxph(Surv(time_surv,event_surv)~age_at_diagnosis+stage_n+meta_n+recur_n,data=d)
    as.data.frame(round(summary(cox)$coefficients,4))
  })
  output$cox_out <- renderPrint({ print(cox_res_df()) })
  register_table_dl(output,"dl_cox_csv", cox_res_df, "cox_ph")

  # ==============================================
  # TRAINING TIMES
  # ==============================================
  time_bar_p <- reactive({
    req(ml())
    td <- ml()$timing %>% arrange(desc(Seconds))
    td$Model <- factor(td$Model, levels=rev(td$Model))
    ggplot(td,aes(x=Model,y=Seconds,fill=Type))+
      geom_bar(stat="identity",alpha=0.88,color="white",width=0.65)+
      geom_text(aes(label=paste0(Seconds,"s")),hjust=-0.1,size=3.5,fontface="bold")+
      scale_fill_manual(values=c("WhiteBox"="#1565C0","BlackBox"="#B71C1C"))+
      coord_flip()+
      scale_y_continuous(expand=expansion(mult=c(0,0.2)))+
      labs(title="Training Time by Model",
           subtitle=paste("Seed:",ml()$params$seed,
                          "| CV:",ml()$params$cv_method,
                          "k=",ml()$params$cv_folds,
                          "| Balance:",ml()$params$balance_method),
           x=NULL,y="Elapsed (seconds)",fill="Type")+
      theme_classic(base_size=12)+theme(legend.position="top",
                                        plot.title=element_text(face="bold"))
  })
  output$plt_time_bar <- renderPlot({ time_bar_p() })
  register_plot_dl(output,"dl_time_bar_png", time_bar_p, "training_time_bar")

  time_auc_p <- reactive({
    req(ml(), all_res())
    mrg <- inner_join(ml()$timing, all_res() %>% select(Model,AUC), by="Model")
    ggplot(mrg,aes(x=Seconds,y=AUC,color=Type,label=Model))+
      geom_point(size=5,alpha=0.85)+
      geom_text_repel(size=3.2,show.legend=FALSE,max.overlaps=20)+
      scale_color_manual(values=c("WhiteBox"="#1565C0","BlackBox"="#B71C1C"))+
      geom_hline(yintercept=0.7,lty=2,color="gray60")+
      labs(title="Time vs AUC Trade-off",
           x="Training Time (sec)",y="Test AUC",color="Type")+
      theme_classic(base_size=12)+theme(legend.position="top")
  })
  output$plt_time_auc <- renderPlot({ time_auc_p() })
  register_plot_dl(output,"dl_time_auc_png", time_auc_p, "time_vs_auc")

  output$dl_time_dot_png <- downloadHandler(
    filename=function() paste0("training_dot_",format(Sys.time(),"%Y%m%d_%H%M%S"),".png"),
    content=function(file) {
      req(ml())
      td <- ml()$timing %>% arrange(Seconds)
      td$Model <- factor(td$Model,levels=td$Model)
      p <- ggplot(td,aes(x=Seconds,y=Model,color=Type,size=Seconds))+
        geom_point(alpha=0.85)+
        scale_color_manual(values=c("WhiteBox"="#1565C0","BlackBox"="#B71C1C"))+
        scale_size_continuous(range=c(4,12))+
        labs(title="Training Time Dot Plot",x="Seconds",y=NULL)+
        theme_classic(base_size=12)+theme(legend.position="top")
      png(file,10,6,units="in",res=600); print(p); dev.off()
    }
  )

  timing_full <- reactive({
    req(ml(),all_res())
    inner_join(ml()$timing, all_res() %>% select(Model,AUC,F1), by="Model") %>%
      mutate(AUC_per_sec=round(AUC/Seconds,4)) %>%
      arrange(desc(AUC))
  })
  output$tbl_timing <- DT::renderDataTable({
    req(timing_full())
    DT::datatable(timing_full(), rownames=FALSE,
      options=list(dom="t",pageLength=15)) %>%
      DT::formatStyle("Type",
        backgroundColor=DT::styleEqual(c("WhiteBox","BlackBox"),c("#E3F2FD","#FFEBEE")))
  })
  register_table_dl(output,"dl_time_csv", timing_full, "training_times")

  # ==============================================
  # WB: LASSO
  # ==============================================
  lasso_coef <- reactive({
    req(ml())
    mdl <- ml()$models$LASSO; bl <- mdl$bestTune$lambda
    cs  <- coef(mdl$finalModel, s=bl)
    data.frame(Variable=rownames(cs),
               Coefficient=round(as.numeric(cs),4),
               OddsRatio=round(exp(as.numeric(cs)),4)) %>%
      filter(Variable!="(Intercept)",Coefficient!=0) %>%
      arrange(desc(abs(Coefficient)))
  })
  output$tbl_lasso_coef <- DT::renderDataTable({
    DT::datatable(lasso_coef(), rownames=FALSE, options=list(dom="t",pageLength=20))
  })
  register_table_dl(output,"dl_lasso_coef_csv", lasso_coef, "lasso_coefficients")

  lasso_coef_p <- reactive({
    df_c <- lasso_coef() %>%
      mutate(Dir=ifelse(Coefficient>0,"Positive","Negative"),
             Variable=factor(Variable,levels=Variable[order(Coefficient)]))
    ggplot(df_c,aes(x=Variable,y=Coefficient,fill=Dir))+
      geom_bar(stat="identity",alpha=0.85)+geom_hline(yintercept=0,lty=2)+
      scale_fill_manual(values=c("Positive"="#D32F2F","Negative"="#1565C0"))+
      coord_flip()+labs(title="LASSO Coefficients",x=NULL,y="Log-Odds")+
      theme_classic(base_size=11)+theme(legend.position="top")
  })
  output$plt_lasso_coef <- renderPlot({ lasso_coef_p() })
  register_plot_dl(output,"dl_lasso_coef_png", lasso_coef_p, "lasso_coef")

  output$plt_lasso_path <- renderPlot({
    req(ml())
    plot(ml()$models$LASSO$finalModel,xvar="lambda",label=TRUE,
         main="LASSO Regularization Path")
    abline(v=log(ml()$models$LASSO$bestTune$lambda),col="red",lty=2,lwd=2)
  })
  output$dl_lasso_path_png <- downloadHandler(
    filename=function() paste0("lasso_path_",format(Sys.time(),"%Y%m%d_%H%M%S"),".png"),
    content=function(file) {
      req(ml())
      png(file,10,8,units="in",res=600)
      plot(ml()$models$LASSO$finalModel,xvar="lambda",label=TRUE,
           main="LASSO Regularization Path")
      abline(v=log(ml()$models$LASSO$bestTune$lambda),col="red",lty=2,lwd=2)
      dev.off()
    }
  )
  output$txt_lasso_eq <- renderText({
    req(ml())
    mdl <- ml()$models$LASSO; bl <- mdl$bestTune$lambda
    cs  <- coef(mdl$finalModel,s=bl)
    df_c <- data.frame(Variable=rownames(cs),Coefficient=round(as.numeric(cs),4)) %>%
      filter(Coefficient!=0)
    intercept <- df_c$Coefficient[df_c$Variable=="(Intercept)"]
    preds <- df_c %>% filter(Variable!="(Intercept)")
    paste0("LASSO (alpha=",ml()$params$lasso_alpha,
           ", lambda=",round(bl,5),")\n\n",
           "logit(Recurrence) =\n  ",round(intercept,4)," +\n  ",
           paste0(round(preds$Coefficient,4)," * ",preds$Variable,collapse=" +\n  "),
           "\n\nP(Recurrence) = 1 / (1 + exp(-logit))")
  })

  # ==============================================
  # WB: CART
  # ==============================================
  output$plt_cart_tree <- renderPlot({
    req(ml())
    rpart.plot(ml()$models$CART$finalModel,type=4,extra=104,
               roundint=FALSE,fallen.leaves=TRUE,
               main="CART Decision Tree",cex=0.75,box.palette="RdYlGn")
  })
  output$dl_cart_tree_png <- downloadHandler(
    filename=function() paste0("cart_tree_",format(Sys.time(),"%Y%m%d_%H%M%S"),".png"),
    content=function(file) {
      req(ml())
      png(file,14,10,units="in",res=600)
      rpart.plot(ml()$models$CART$finalModel,type=4,extra=104,roundint=FALSE,
                 fallen.leaves=TRUE,main="CART Decision Tree",cex=0.75,box.palette="RdYlGn")
      dev.off()
    }
  )
  output$txt_cart_rules <- renderPrint({ req(ml()); rpart.rules(ml()$models$CART$finalModel,cover=TRUE) })

  cart_vi_df <- reactive({
    req(ml())
    varImp(ml()$models$CART)$importance %>% as.data.frame() %>%
      rownames_to_column("Variable") %>% filter(Overall>0) %>% arrange(desc(Overall))
  })
  cart_vi_p <- reactive({
    df_v <- cart_vi_df() %>% mutate(Variable=factor(Variable,levels=rev(Variable)))
    ggplot(df_v,aes(x=Variable,y=Overall,fill=Overall))+
      geom_bar(stat="identity")+coord_flip()+
      scale_fill_gradient(low="#90CAF9",high="#1565C0")+
      labs(title="CART - Gini Importance",x=NULL,y="Importance")+
      theme_classic(base_size=11)+theme(legend.position="none")
  })
  output$plt_cart_vi <- renderPlot({ cart_vi_p() })
  register_plot_dl(output,"dl_cart_vi_png", cart_vi_p, "cart_varimp")
  register_table_dl(output,"dl_cart_vi_csv", cart_vi_df, "cart_varimp")

  cart_cm_p <- reactive({
    req(ml()); m <- ml()
    pred <- predict(m$models$CART, m$test_df)
    as.data.frame(confusionMatrix(pred,m$test_df$recurrence)$table) %>%
      ggplot(aes(x=Reference,y=Prediction,fill=Freq))+
      geom_tile(color="white")+geom_text(aes(label=Freq),size=8,fontface="bold")+
      scale_fill_gradient(low="white",high="#1565C0")+
      labs(title="CART Confusion Matrix")+theme_classic()+theme(legend.position="none")
  })
  output$plt_cart_cm <- renderPlot({ cart_cm_p() })
  register_plot_dl(output,"dl_cart_cm_png", cart_cm_p, "cart_cm")

  # ==============================================
  # WB: NB
  # ==============================================
  output$txt_nb_prior <- renderPrint({
    req(ml()); mdl <- ml()$models$NaiveBayes$finalModel
    cat("Prior Probabilities:\n"); print(round(mdl$apriori/sum(mdl$apriori),4))
  })
  nb_lik_p <- reactive({
    req(ml()); m <- ml(); feat <- input$nb_feat
    d <- m$train_bal %>% mutate(fv=as.numeric(.data[[feat]])) %>% filter(!is.na(fv))
    ggplot(d,aes(x=fv,fill=recurrence))+geom_density(alpha=0.6,color="white")+
      scale_fill_manual(values=c("Rec0"="#1565C0","Rec1"="#D32F2F"),
                        labels=c("No Recurrence","Recurrence"))+
      labs(title=paste("P(",feat,"| Class)"),x=feat,y="Density",fill="Class")+
      theme_classic(base_size=12)
  })
  output$plt_nb_lik <- renderPlot({ nb_lik_p() })
  register_plot_dl(output,"dl_nb_lik_png", nb_lik_p, "nb_likelihood")

  # ==============================================
  # WB: KNN
  # ==============================================
  knn_k_p <- reactive({
    req(ml())
    dr <- ml()$models$KNN$results
    ggplot(dr,aes(x=k,y=ROC))+geom_line(color="#1565C0",lwd=1.2)+
      geom_point(color="#1565C0",size=3)+
      geom_vline(xintercept=ml()$models$KNN$bestTune$k,lty=2,color="red")+
      labs(title="KNN - CV ROC vs k",x="k",y="CV AUC")+theme_classic(base_size=12)
  })
  output$plt_knn_k <- renderPlot({ knn_k_p() })
  register_plot_dl(output,"dl_knn_k_png", knn_k_p, "knn_k")

  knn_pca_p <- reactive({
    req(ml()); m <- ml()
    pca <- prcomp(m$train_bal %>% select(-recurrence),scale.=TRUE)
    data.frame(PC1=pca$x[,1],PC2=pca$x[,2],Class=m$train_bal$recurrence) %>%
      ggplot(aes(x=PC1,y=PC2,color=Class))+geom_point(alpha=0.6,size=2)+
      scale_color_manual(values=c("Rec0"="#1565C0","Rec1"="#D32F2F"))+
      stat_ellipse(aes(group=Class),lwd=1.2)+
      labs(title="KNN - PCA Projection",x="PC1",y="PC2")+theme_classic(base_size=12)
  })
  output$plt_knn_pca <- renderPlot({ knn_pca_p() })
  register_plot_dl(output,"dl_knn_pca_png", knn_pca_p, "knn_pca")

  # ==============================================
  # BB: PERFORMANCE
  # ==============================================
  output$tbl_perf <- DT::renderDataTable({ req(all_res()); fmt_dt(all_res()) })
  register_table_dl(output,"dl_perf_csv", all_res, "model_performance")

  auc_bar_p <- reactive({
    req(all_res())
    res <- all_res() %>% arrange(AUC) %>% mutate(Model=factor(Model,levels=Model))
    ggplot(res,aes(x=Model,y=AUC,fill=Type))+
      geom_bar(stat="identity",alpha=0.85,color="white")+
      geom_text(aes(label=AUC),hjust=-0.1,size=3.5)+
      scale_fill_manual(values=c("WhiteBox"="#1565C0","BlackBox"="#B71C1C"))+
      coord_flip()+ylim(0,1.12)+labs(title="AUC by Model",x=NULL,y="Test AUC")+
      theme_classic(base_size=12)
  })
  output$plt_auc_bar <- renderPlot({ auc_bar_p() })
  register_plot_dl(output,"dl_auc_bar_png", auc_bar_p, "auc_comparison")

  cv_box_p <- reactive({
    req(ml())
    m <- ml()
    all_caret <- m$models[setdiff(names(m$models),"XGBoost")]
    cv_res  <- resamples(all_caret)
    cv_res$values %>% select(ends_with("ROC")) %>%
      pivot_longer(everything(),names_to="Model",values_to="AUC") %>%
      mutate(Model=gsub("~ROC$","",Model),
             Type=ifelse(Model %in% m$whitebox,"WhiteBox","BlackBox")) %>%
      ggplot(aes(x=Model,y=AUC,fill=Type))+
      geom_boxplot(alpha=0.75)+geom_jitter(width=0.12,alpha=0.3,size=1)+
      scale_fill_manual(values=c("WhiteBox"="#1565C0","BlackBox"="#B71C1C"))+
      labs(title="CV AUC Distribution",x=NULL,y="AUC")+
      theme_classic(base_size=11)+theme(axis.text.x=element_text(angle=30,hjust=1))
  })
  output$plt_cv_box <- renderPlot({ cv_box_p() })
  register_plot_dl(output,"dl_cv_box_png", cv_box_p, "cv_distribution")

  # ==============================================
  # SHAP
  # ==============================================
  shap_data <- reactive({
    req(ml()); m <- ml()
    x_test <- as.matrix(m$test_df[,m$xgb_feat])
    sv <- predict(m$models$XGBoost, xgb.DMatrix(x_test), predcontrib=TRUE)
    sv <- sv[,colnames(sv)!="BIAS"]
    list(shap=sv, x=x_test)
  })

  shap_sum_p <- reactive({
    sv <- shap_data()
    sl <- as.data.frame(sv$shap) %>% mutate(obs=row_number()) %>%
      pivot_longer(-obs,names_to="Feature",values_to="SHAP") %>%
      left_join(as.data.frame(sv$x) %>% mutate(obs=row_number()) %>%
                  pivot_longer(-obs,names_to="Feature",values_to="Value"),
                by=c("obs","Feature")) %>%
      group_by(Feature) %>% mutate(MeanAbs=mean(abs(SHAP))) %>% ungroup() %>%
      mutate(Feature=reorder(Feature,MeanAbs))
    ggplot(sl,aes(x=Feature,y=SHAP,color=Value))+
      geom_jitter(width=0.2,alpha=0.7,size=2)+coord_flip()+
      scale_color_gradient(low="#1565C0",high="#D32F2F",name="Feature\nValue")+
      geom_hline(yintercept=0,lty=2,color="gray50")+
      labs(title="SHAP Summary (XGBoost)",x=NULL,y="SHAP Value")+
      theme_classic(base_size=11)
  })
  output$plt_shap_sum <- renderPlot({ shap_sum_p() })
  register_plot_dl(output,"dl_shap_sum_png", shap_sum_p, "shap_summary")

  shap_imp_df <- reactive({
    sv <- shap_data()
    colMeans(abs(sv$shap)) %>% sort(decreasing=TRUE) %>%
      as.data.frame() %>% setNames("MeanAbsSHAP") %>% rownames_to_column("Feature")
  })
  shap_imp_p <- reactive({
    df_s <- shap_imp_df() %>% mutate(Feature=factor(Feature,levels=rev(Feature)))
    ggplot(df_s,aes(x=Feature,y=MeanAbsSHAP,fill=MeanAbsSHAP))+
      geom_bar(stat="identity")+coord_flip()+
      scale_fill_gradient(low="#90CAF9",high="#1565C0")+
      labs(title="Mean |SHAP| Importance",x=NULL,y="Mean |SHAP|")+
      theme_classic(base_size=11)+theme(legend.position="none")
  })
  output$plt_shap_imp <- renderPlot({ shap_imp_p() })
  register_plot_dl(output,"dl_shap_imp_png", shap_imp_p, "shap_importance")
  register_table_dl(output,"dl_shap_imp_csv", shap_imp_df, "shap_importance")

  shap_wf_p <- reactive({
    sv <- shap_data(); m <- ml()
    obs_i <- min(input$shap_obs, nrow(sv$shap))
    si <- sv$shap[obs_i,]
    bs <- mean(predict(m$models$XGBoost, m$xgb_te))
    data.frame(Feature=names(si),SHAP=as.numeric(si)) %>%
      arrange(desc(abs(SHAP))) %>% head(10) %>%
      mutate(Dir=ifelse(SHAP>0,"Increases Risk","Decreases Risk"),
             Feature=factor(Feature,levels=rev(Feature))) %>%
      ggplot(aes(x=Feature,y=SHAP,fill=Dir))+
      geom_bar(stat="identity",alpha=0.85)+geom_hline(yintercept=0,lty=2)+
      coord_flip()+
      scale_fill_manual(values=c("Increases Risk"="#D32F2F","Decreases Risk"="#1565C0"))+
      labs(title=paste("SHAP Waterfall - Patient",obs_i),
           subtitle=paste("Base score:",round(bs,3)),x=NULL,y="SHAP")+
      theme_classic(base_size=11)+theme(legend.position="top")
  })
  output$plt_shap_wf <- renderPlot({ shap_wf_p() })
  register_plot_dl(output,"dl_shap_wf_png", shap_wf_p, "shap_waterfall")

  # ==============================================
  # DALEX
  # ==============================================
  make_explainer <- function(m, nm) {
    x_test <- m$test_df %>% select(-recurrence)
    y_test <- as.numeric(m$test_df$recurrence=="Rec1")
    if (nm=="XGBoost") {
      DALEX::explain(m$models$XGBoost, data=x_test[,m$xgb_feat], y=y_test,
                     predict_function=function(model,nd)
                       predict(model,xgb.DMatrix(as.matrix(nd))),
                     label="XGBoost", verbose=FALSE)
    } else {
      DALEX::explain(m$models[[nm]], data=x_test, y=y_test, label=nm, verbose=FALSE)
    }
  }

  dalex_bd_res <- eventReactive(input$btn_dalex, {
    req(ml()); m <- ml(); nm <- input$dalex_mdl
    exp <- make_explainer(m, nm)
    obs_i <- min(input$dalex_obs, nrow(m$test_df))
    obs_df <- if (nm=="XGBoost") m$test_df[obs_i, m$xgb_feat, drop=FALSE]
              else m$test_df[obs_i, setdiff(names(m$test_df),"recurrence"), drop=FALSE]
    predict_parts(exp, new_observation=obs_df, type="break_down")
  })
  dalex_bd_p <- reactive({
    req(dalex_bd_res())
    plot(dalex_bd_res())+
      labs(title=paste("DALEX Break-Down -",input$dalex_mdl,"| Patient",input$dalex_obs))+
      theme_classic(base_size=11)
  })
  output$plt_dalex_bd <- renderPlot({ dalex_bd_p() })
  register_plot_dl(output,"dl_dalex_bd_png", dalex_bd_p, "dalex_breakdown")

  dalex_vi_res <- eventReactive(input$btn_dalex_vi, {
    req(ml()); m <- ml(); nm <- input$dalex_vi_mdl
    exp <- make_explainer(m, nm)
    model_parts(exp, type="variable_importance", B=10, loss_function=loss_one_minus_auc)
  })
  dalex_vi_p <- reactive({
    req(dalex_vi_res())
    plot(dalex_vi_res())+
      labs(title=paste("Permutation Importance -",input$dalex_vi_mdl))+
      theme_classic(base_size=11)
  })
  output$plt_dalex_vi <- renderPlot({ dalex_vi_p() })
  register_plot_dl(output,"dl_dalex_vi_png", dalex_vi_p, "dalex_varimp")

  # ==============================================
  # VARIABLE IMPORTANCE
  # ==============================================
  vi_xgb_df <- reactive({
    req(ml()); xgb.importance(feature_names=ml()$xgb_feat,
                               model=ml()$models$XGBoost) %>% as.data.frame()
  })
  vi_xgb_p <- reactive({
    df_v <- vi_xgb_df() %>% head(12) %>%
      mutate(Feature=factor(Feature,levels=rev(Feature[order(Gain)])))
    ggplot(df_v,aes(x=Feature,y=Gain,fill=Gain))+geom_bar(stat="identity")+coord_flip()+
      scale_fill_gradient(low="#90CAF9",high="#1565C0")+
      labs(title="XGBoost - Gain",x=NULL,y="Gain")+
      theme_classic(base_size=11)+theme(legend.position="none")
  })
  output$plt_vi_xgb <- renderPlot({ vi_xgb_p() })
  register_plot_dl(output,"dl_vi_xgb_png", vi_xgb_p, "xgb_varimp")
  register_table_dl(output,"dl_vi_xgb_csv", vi_xgb_df, "xgb_varimp")

  vi_rf_df <- reactive({
    req(ml()); varImp(ml()$models$RandomForest)$importance %>%
      as.data.frame() %>% rownames_to_column("Variable") %>%
      arrange(desc(Overall)) %>% head(12)
  })
  vi_rf_p <- reactive({
    df_v <- vi_rf_df() %>% mutate(Variable=factor(Variable,levels=rev(Variable)))
    ggplot(df_v,aes(x=Variable,y=Overall,fill=Overall))+geom_bar(stat="identity")+coord_flip()+
      scale_fill_gradient(low="#A5D6A7",high="#1B5E20")+
      labs(title="Random Forest - Gini",x=NULL,y="Importance")+
      theme_classic(base_size=11)+theme(legend.position="none")
  })
  output$plt_vi_rf <- renderPlot({ vi_rf_p() })
  register_plot_dl(output,"dl_vi_rf_png", vi_rf_p, "rf_varimp")
  register_table_dl(output,"dl_vi_rf_csv", vi_rf_df, "rf_varimp")

  vi_gbm_df <- reactive({
    req(ml()); summary(ml()$models$GBM$finalModel,plotit=FALSE) %>%
      as.data.frame() %>% head(12)
  })
  vi_gbm_p <- reactive({
    df_v <- vi_gbm_df() %>% mutate(var=factor(var,levels=rev(var)))
    ggplot(df_v,aes(x=var,y=rel.inf,fill=rel.inf))+geom_bar(stat="identity")+coord_flip()+
      scale_fill_gradient(low="#FFCC80",high="#E65100")+
      labs(title="GBM - Relative Influence",x=NULL,y="Rel. Influence (%)")+
      theme_classic(base_size=11)+theme(legend.position="none")
  })
  output$plt_vi_gbm <- renderPlot({ vi_gbm_p() })
  register_plot_dl(output,"dl_vi_gbm_png", vi_gbm_p, "gbm_varimp")
  register_table_dl(output,"dl_vi_gbm_csv", vi_gbm_df, "gbm_varimp")

  pdp_res <- eventReactive(input$btn_pdp, {
    req(ml()); m <- ml(); nm <- input$pdp_mdl
    x_test <- m$test_df %>% select(-recurrence)
    y_test <- as.numeric(m$test_df$recurrence=="Rec1")
    exp <- DALEX::explain(m$models[[nm]], data=x_test, y=y_test, label=nm, verbose=FALSE)
    model_profile(exp, variables=input$pdp_feat, type="partial")
  })
  pdp_p <- reactive({
    req(pdp_res())
    plot(pdp_res())+labs(title=paste("PDP -",input$pdp_feat,"(",input$pdp_mdl,")"))+
      theme_classic(base_size=11)
  })
  output$plt_pdp <- renderPlot({ pdp_p() })
  register_plot_dl(output,"dl_pdp_png", pdp_p, "partial_dependence")

  # ==============================================
  # ROC
  # ==============================================
  roc_fn <- reactive({
    req(ml()); m <- ml()
    pal <- c("#1565C0","#388E3C","#F57C00","#7B1FA2",
             "#B71C1C","#00838F","#6A1B9A","#2E7D32","#37474F")
    all_nms <- c(m$whitebox, setdiff(m$blackbox,"XGBoost"),"XGBoost")
    roc_list <- lapply(all_nms, function(nm) {
      prob <- if (nm=="XGBoost") predict(m$models$XGBoost,m$xgb_te)
              else predict(m$models[[nm]],m$test_df,type="prob")[,"Rec1"]
      pROC::roc(as.numeric(m$test_df$recurrence=="Rec1"),as.numeric(prob),quiet=TRUE)
    })
    names(roc_list) <- all_nms
    aucs <- sapply(roc_list,function(r) round(as.numeric(pROC::auc(r)),3))
    list(roc_list=roc_list, aucs=aucs, pal=pal)
  })

  draw_roc <- function(rv) {
    plot(rv$roc_list[[1]],col=rv$pal[1],lwd=2.5,
         main="ROC Curves - All 9 Models",
         xlab="1 - Specificity",ylab="Sensitivity",cex.main=1.3)
    for (i in 2:length(rv$roc_list))
      plot(rv$roc_list[[i]],col=rv$pal[i],lwd=2.5,add=TRUE)
    abline(a=0,b=1,lty=2,col="gray60")
    legend("bottomright",
           legend=sprintf("%s (AUC=%.3f)",names(rv$aucs),rv$aucs),
           col=rv$pal[seq_along(rv$aucs)],lwd=2.5,bty="n",cex=0.82)
  }
  output$plt_roc <- renderPlot({ draw_roc(roc_fn()) })
  output$dl_roc_png <- downloadHandler(
    filename=function() paste0("roc_",format(Sys.time(),"%Y%m%d_%H%M%S"),".png"),
    content=function(file) {
      req(ml()); png(file,10,8,units="in",res=600); draw_roc(roc_fn()); dev.off()
    }
  )
  auc_df <- reactive({ req(all_res()); all_res()[,c("Model","Type","AUC","F1")] })
  output$tbl_auc <- DT::renderDataTable({ fmt_dt(auc_df()) })
  register_table_dl(output,"dl_auc_csv", auc_df, "auc_table")

  # ==============================================
  # PREDICTION TOOL
  # ==============================================
  new_pt <- reactive({
    data.frame(age_at_diagnosis=as.numeric(input$pi_age),
               sex=as.numeric(input$pi_sex), disease_type=as.numeric(input$pi_dtype),
               ret_mutation=0, tumor_size_mm=as.numeric(input$pi_tumor),
               stage=as.numeric(input$pi_stage), multifocal=0,
               lymph_node_invasion=as.numeric(input$pi_lni),
               capsular_invasion=0, soft_tissue_invasion=0,
               metastasis_at_diagnosis=as.numeric(input$pi_meta),
               calcitonin_preop=as.numeric(input$pi_cal),
               cea_preop=as.numeric(input$pi_cea),
               lymph_node_metastasis=as.numeric(input$pi_lni),
               lung_metastasis_present=0, bone_metastasis_present=0,
               liver_metastasis_present=0)
  })

  pred_probs <- eventReactive(input$btn_pred, {
    req(ml()); m <- ml(); pt <- new_pt()
    caret_nms <- setdiff(names(m$models),"XGBoost")
    probs <- sapply(caret_nms, function(nm)
      tryCatch(predict(m$models[[nm]],pt,type="prob")[,"Rec1"],
               error=function(e) NA_real_))
    xgb_p <- predict(m$models$XGBoost,
                     xgb.DMatrix(as.matrix(pt[,m$xgb_feat])))
    c(probs, XGBoost=xgb_p)
  })

  pred_bar_p <- reactive({
    req(pred_probs())
    pp <- pred_probs()
    data.frame(Model=names(pp),Prob=as.numeric(pp)) %>% filter(!is.na(Prob)) %>%
      mutate(Risk=ifelse(Prob>=0.5,"High Risk","Low Risk"),
             Model=factor(Model,levels=Model[order(Prob)])) %>%
      ggplot(aes(x=Model,y=Prob,fill=Risk))+
      geom_bar(stat="identity",alpha=0.85)+
      geom_hline(yintercept=0.5,lty=2,lwd=1)+coord_flip()+
      scale_fill_manual(values=c("High Risk"="#D32F2F","Low Risk"="#1565C0"))+
      scale_y_continuous(limits=c(0,1),labels=scales::percent)+
      labs(title="Recurrence Probability per Model",x=NULL,y="P(Recurrence)")+
      theme_classic(base_size=12)
  })
  output$plt_pred_bar <- renderPlot({ pred_bar_p() })
  register_plot_dl(output,"dl_pred_bar_png", pred_bar_p, "prediction_probabilities")

  output$ui_consensus <- renderUI({
    req(pred_probs())
    pp <- pred_probs(); high <- sum(pp>=0.5,na.rm=TRUE)
    total <- sum(!is.na(pp)); avg <- round(mean(pp,na.rm=TRUE)*100,1)
    col <- if (avg>=50) "#B71C1C" else "#1565C0"; txt <- if (avg>=50) "HIGH RISK" else "LOW RISK"
    tagList(div(style=paste0("background:",col,";color:white;padding:16px;",
                              "border-radius:8px;text-align:center;"),
                h3(txt), h4(paste0("Mean P = ",avg,"%")),
                p(paste0(high,"/",total," models vote RECURRENCE"))))
  })

  pred_exp_p <- reactive({
    req(pred_probs(), ml()); m <- ml()
    x_test <- m$test_df %>% select(-recurrence)
    y_test  <- as.numeric(m$test_df$recurrence=="Rec1")
    exp_rf  <- DALEX::explain(m$models$RandomForest, data=x_test, y=y_test,
                               label="RF", verbose=FALSE)
    bd <- predict_parts(exp_rf, new_observation=new_pt(), type="break_down")
    plot(bd)+labs(title="DALEX Explanation (Random Forest)")+theme_classic(base_size=10)
  })
  output$plt_pred_exp <- renderPlot({ pred_exp_p() })
  register_plot_dl(output,"dl_pred_exp_png", pred_exp_p, "prediction_explanation")
}

shinyApp(ui=ui, server=server)

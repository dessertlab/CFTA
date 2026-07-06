suppressWarnings({
  library(readr)
  library(fs)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(ggplot2)
  library(faircause)
  library(progress)
})

set.seed(42)

local({
  args <- commandArgs(trailingOnly = FALSE)
  sp <- sub("^--file=", "", args[grep("^--file=", args)])
  if (!length(sp)) sp <- tryCatch(normalizePath(sys.frames()[[1]]$ofile),
                                   error = function(e) NA_character_)
  start <- if (length(sp) && !is.na(sp)) dirname(normalizePath(sp)) else getwd()
  root <- normalizePath(start, winslash = "/", mustWork = FALSE)
  while (!file.exists(file.path(root, "CODE", "common", "r_paths.R"))) {
    parent <- dirname(root)
    if (parent == root) stop("Could not locate CODE/common/r_paths.R above: ", start)
    root <- parent
  }
  source(file.path(root, "CODE", "common", "r_paths.R"))
  rp_init(root)
})

ATTR      <- rp_resolve_attr(default = "RACE")
GENERATOR <- rp_resolve_generator(default = "CHATGPT4")
scm       <- rp_scm(ATTR)

base_sa <- rp_sa_dir(ATTR, GENERATOR)
out_rq3 <- rp_rq3_dir(ATTR, GENERATOR)
dir_create(out_rq3, recurse = TRUE)

scenarios <- rp_scenarios(ATTR)
y_candidates <- RP_Y_CANDIDATES
budgets_per_scen <- c(100,200,400,600,800)
age_breaks <- c(-Inf,25,35,45,55,65,Inf)

log_line <- function(level, ...) {
  msg <- sprintf("[%s] %s | %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), level, paste0(...))
  cat(msg, "\n"); flush.console()
}
log_info <- function(...) log_line("INFO", ...)
log_warn <- function(...) log_line("WARN", ...)
log_err  <- function(...) log_line("ERROR", ...)

log_info(ATTR, " / ", GENERATOR, " | RQ3 stratified sensitivity")
log_info("Input : ", base_sa)
if (!dir_exists(base_sa)) { log_err("Input folder not found: ", base_sa, " (run the sentiment analysis first)"); quit(status = 1) }

find_scenario_csv <- function(dir_model, s) {
  pat <- paste0("(^|_)s", sub("^s","",s), ".*\\.csv$",
                "|generated_sentences_", s, ".*\\.csv$",
                "|dataset_with_sentiment_scores_.*_", s, "\\.csv$")
  p <- dir_ls(dir_model, regexp = pat, type = "file")
  if (length(p) > 0) p[1] else character(0)
}

run_decomposition <- function(df, ycol) {
  if ("Prior Convictions" %in% names(df)) {
    names(df)[names(df) == "Prior Convictions"] <- "Prior.Convictions"
  }
  required <- c(scm$X, scm$Z, scm$W, ycol)
  if (!all(required %in% names(df))) return(NULL)
  suppressWarnings({
    fairness_cookbook(
      data   = df,
      X      = scm$X,
      Z      = scm$Z,
      W      = scm$W,
      Y      = ycol,
      x0     = scm$x0,
      x1     = scm$x1,
      method = "debiasing"
    )
  })
}

extract_estimates <- function(res_obj) {
  p <- tryCatch(autoplot(res_obj), error = function(e) NULL)
  if (!is.null(p)) {
    dfp <- tryCatch(p$data, error = function(e) NULL)
    if (!is.null(dfp) && all(c("Effect","Estimate") %in% names(dfp))) {
      out <- dfp %>% dplyr::select(Effect, Estimate)
      if (any(is.finite(out$Estimate))) return(out)
    }
    gb <- tryCatch(ggplot_build(p), error = function(e) NULL)
    if (!is.null(gb) && length(gb$data) >= 1) {
      lyr <- gb$data[[1]]
      if ("y" %in% names(lyr)) {
        default_effects <- c("Total","Direct","Indirect","Spurious")
        k <- min(length(default_effects), nrow(lyr))
        out <- tibble::tibble(
          Effect   = default_effects[seq_len(k)],
          Estimate = as.numeric(lyr$y[seq_len(k)])
        )
        if (any(is.finite(out$Estimate))) return(out)
      }
    }
  }
  df2 <- tryCatch(as.data.frame(res_obj), error=function(e) NULL)
  if (!is.null(df2)) {
    candE <- intersect(c("Effect","effect","component","name"), names(df2))
    candV <- intersect(c("Estimate","estimate","value","est"), names(df2))
    if (length(candE)==1 && length(candV)==1) {
      out <- df2[, c(candE, candV)]
      names(out) <- c("Effect","Estimate")
      out$Effect <- as.character(out$Effect)
      out$Estimate <- as.numeric(out$Estimate)
      if (any(is.finite(out$Estimate))) return(tibble::as_tibble(out))
    }
  }
  slots <- c("total","direct","indirect","spurious","tau_total","tau_direct","tau_indirect","tau_spurious")
  vals <- lapply(slots, function(nm) tryCatch(as.numeric(res_obj[[nm]]), error=function(e) NA_real_))
  names(vals) <- slots
  if (any(is.finite(unlist(vals)))) {
    m <- c(
      Total    = dplyr::coalesce(vals$total, vals$tau_total, NA_real_),
      Direct   = dplyr::coalesce(vals$direct, vals$tau_direct, NA_real_),
      Indirect = dplyr::coalesce(vals$indirect, vals$tau_indirect, NA_real_),
      Spurious = dplyr::coalesce(vals$spurious, vals$tau_spurious, NA_real_)
    )
    out <- tibble::tibble(Effect = names(m), Estimate = as.numeric(m))
    if (any(is.finite(out$Estimate))) return(out)
  }
  return(NULL)
}

make_strata <- function(df) {
  if ("Prior Convictions" %in% names(df)) {
    names(df)[names(df) == "Prior Convictions"] <- "Prior.Convictions"
  }
  df %>%
    mutate(
      X  = as.factor(Race),
      W1 = as.factor(Prior.Convictions),
      Zb = cut(Age, breaks = age_breaks, right = FALSE, include.lowest = TRUE)
    ) %>%
    tidyr::unite("STRATA", X, W1, Zb, remove = FALSE, sep = " | ")
}

ref_distribution <- function(df_str) {
  df_str %>% count(STRATA, name = "N_ref") %>% mutate(p_ref = N_ref / sum(N_ref))
}

alloc_counts <- function(n_total, ref_dist, avail_counts) {
  if (n_total <= 0) return(tibble(STRATA=character(0), n_take=integer(0)))
  target <- ref_dist %>% mutate(n_target = floor(n_total * p_ref))
  rem <- n_total - sum(target$n_target)
  if (rem > 0) {
    idx <- order(ref_dist$p_ref, decreasing = TRUE)[seq_len(rem)]
    target$n_target[idx] <- target$n_target[idx] + 1
  }
  target$n_take <- pmin(target$n_target, avail_counts[match(target$STRATA, names(avail_counts))])
  leftover <- n_total - sum(target$n_take)
  if (leftover > 0) {
    caps <- avail_counts[match(target$STRATA, names(avail_counts))] - target$n_take
    while (leftover > 0 && any(caps > 0)) {
      j <- which.max(ref_dist$p_ref * (caps > 0))
      target$n_take[j] <- target$n_take[j] + 1
      caps[j] <- caps[j] - 1
      leftover <- leftover - 1
    }
  }
  target %>% select(STRATA, n_take)
}

stratified_sample_n <- function(df_str, n_total, ref_dist) {
  if (n_total <= 0) return(df_str[0,])
  avail <- df_str %>% count(STRATA, name = "n")
  avail_counts <- setNames(avail$n, avail$STRATA)
  alloc <- alloc_counts(n_total, ref_dist, avail_counts)
  purrr::map_dfr(seq_len(nrow(alloc)), function(i){
    h <- alloc$STRATA[i]; k <- alloc$n_take[i]
    if (is.na(k) || k <= 0) return(df_str[0,])
    rows_h <- which(df_str$STRATA == h)
    df_str[sample(rows_h, size = min(k, length(rows_h)), replace = FALSE), , drop = FALSE]
  })
}

check_stratification <- function(df_orig, df_sub, vars = c("Race","Prior.Convictions","Zb")) {
  diffs <- sapply(vars, function(v) {
    p_orig <- prop.table(table(df_orig[[v]]))
    p_sub  <- prop.table(table(df_sub[[v]]))
    all_levels <- union(names(p_orig), names(p_sub))
    p_orig <- p_orig[all_levels]; p_sub <- p_sub[all_levels]
    p_orig[is.na(p_orig)] <- 0; p_sub[is.na(p_sub)] <- 0
    max(abs(p_orig - p_sub))
  })
  paste0(names(diffs), " Δmax=", sprintf("%.3f", diffs), collapse=" | ")
}

short_label <- function(x) {
  xl <- tolower(x)
  dplyr::case_when(
    grepl("^allam[_-]?2[_-]?7b", xl) ~ "allam",
    grepl("^cardiffnlp", xl) ~ "rob_twit",
    grepl("^distilbert", xl) ~ "distill",
    grepl("^gemma2", xl) ~ "gemma",
    grepl("^gpt[_-]?3[_-]?5", xl) ~ "gpt",
    grepl("^llama3[_-]?70b", xl) ~ "llama_small",
    grepl("^llama[_-]?3[_-]?1[_-]?8b", xl) ~ "llama_large",
    grepl("^siebert", xl) ~ "sieb",
    grepl("^textattack", xl) ~ "bert",
    TRUE ~ x
  )
}

models <- dir_ls(base_sa, type = "directory", recurse = FALSE) %>% path_file()
models <- rp_drop_toxbert(models)
log_info("Models found: ", paste(models, collapse = ", "))

all_model_curves <- list()

for (m in models) {
  out_model_dir <- path(out_rq3, "RSTUDIO", m)
  has_files <- dir_exists(out_model_dir) && length(dir_ls(out_model_dir, type = "file")) > 0
  if (has_files) {
    log_info("Skipping model ", m, " because output folder already exists and is not empty: ", out_model_dir)
    next
  }
  log_info("== Start model: ", m)
  dir_create(out_model_dir, recurse = TRUE)
  scen_paths <- purrr::map(scenarios, ~ find_scenario_csv(path(base_sa, m), .x))
  has_file   <- vapply(scen_paths, length, 1L) > 0
  n_tasks    <- sum(has_file) * length(budgets_per_scen)
  if (n_tasks == 0) { log_warn("No scenario files for ", m); next }
  pb <- progress_bar$new(
    format = paste0("  [", m, "] :percent [:bar] :current/:total | :elapsed ETA :eta"),
    total = n_tasks, clear = FALSE, width = 70
  )
  per_scen_rows <- list()
  for (idx in seq_along(scenarios)) {
    s <- scenarios[idx]
    f <- scen_paths[[idx]]
    if (length(f) == 0) { log_warn("Missing file: ", m, " / ", s); next }
    df <- tryCatch(readr::read_csv(f, show_col_types = FALSE), error = function(e) NULL)
    if (is.null(df)) { log_err("Error reading CSV: ", f); next }
    ycol <- intersect(y_candidates, names(df))[1]
    if (is.na(ycol)) { log_warn("No valid Y in ", f); next }
    df <- df %>% filter(!is.na(.data[[ycol]]))
    if (!nrow(df)) { log_warn("Empty dataset after NA drop: ", f); next }
    df_str <- make_strata(df)
    ref    <- ref_distribution(df_str)
    for (b in budgets_per_scen) {
      n_take <- min(b, nrow(df_str))
      df_sub <- stratified_sample_n(df_str, n_total = n_take, ref_dist = ref)
      log_info("     Check distributions: ", check_stratification(df_str, df_sub))
      res <- tryCatch(run_decomposition(df_sub, ycol), error = function(e) NULL)
      est <- if (!is.null(res)) extract_estimates(res) else NULL
      if (!is.null(est)) {
        est$model    <- m
        est$scenario <- s
        est$budget   <- b
        per_scen_rows[[length(per_scen_rows)+1]] <- est
      }
      pb$tick()
    }
  }
  if (!length(per_scen_rows)) { log_warn("No estimates for model ", m); next }
  model_df <- bind_rows(per_scen_rows)
  write_csv(model_df, path(out_model_dir, paste0(m, "_perSCENARIO_stratified.csv")))
  curve_m <- model_df %>%
    group_by(model, budget, Effect) %>%
    summarise(Estimate = mean(Estimate, na.rm = TRUE), .groups = "drop")
  write_csv(curve_m, path(out_model_dir, paste0(m, "_AGG_curves.csv")))
  all_model_curves[[m]] <- curve_m
  log_info("== End model: ", m)
}

agg_all <- bind_rows(all_model_curves)

if (!nrow(agg_all)) {
  log_info("No new curves computed, loading existing *_AGG_curves.csv to regenerate plots.")
  existing <- list()
  for (m in models) {
    f <- path(out_rq3, "RSTUDIO", m, paste0(m, "_AGG_curves.csv"))
    if (file_exists(f)) {
      dfm <- tryCatch(read_csv(f, show_col_types = FALSE), error = function(e) NULL)
      if (!is.null(dfm)) existing[[m]] <- dfm
    }
  }
  if (length(existing)) agg_all <- bind_rows(existing)
}

if (nrow(agg_all)) {
  plot_dir <- path(out_rq3, "PLOTS"); dir_create(plot_dir, recurse = TRUE)
  agg_all$model_short <- vapply(agg_all$model, short_label, character(1))
  eff_levels <- c("Total","Direct","Indirect","Spurious")
  agg_all$Effect <- factor(agg_all$Effect, levels = eff_levels)
  p_all <- ggplot(
      agg_all,
      aes(x = budget, y = Estimate, group = model_short,
          color = model_short, shape = model_short)
    ) +
    geom_line(linewidth = 1.3) +
    geom_point(size = 3.2) +
    scale_x_continuous(breaks = sort(unique(agg_all$budget))) +
    scale_color_brewer(palette = "Paired") +
    scale_shape_manual(values = c(0,1,2,3,4,5,6,7,8,9)) +
    facet_wrap(~Effect, nrow = 1, scales = "free_y") +
    labs(x = "Budget per scenario", y = "Average bias estimate",
         color = "Model", shape = "Model") +
    theme_minimal(base_size = 17) +
    theme(strip.text = element_text(size = 16, face = "bold"),
          plot.title = element_blank(),
          legend.position = "bottom",
          legend.direction = "horizontal",
          legend.title = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.minor = element_blank())
  ggsave(path(plot_dir, paste0("sensitivity_all_effects_", tolower(ATTR), ".pdf")), p_all, width = 15, height = 6.5, device = cairo_pdf)
  write_csv(agg_all, path(out_rq3, "RQ3_stratified_summary_all_models.csv"))
} else {
  log_warn("No curves available to plot.")
}

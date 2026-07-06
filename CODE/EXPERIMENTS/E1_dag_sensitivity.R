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
})

options(warn = 1)
set.seed(42)

now_ts   <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_line <- function(level, ...) { cat(sprintf("[%s] %s | %s\n", now_ts(), level, paste0(...))); flush.console() }
log_info <- function(...) log_line("INFO",  ...)
log_warn <- function(...) log_line("WARN",  ...)
log_err  <- function(...) log_line("ERROR", ...)

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

ATTR      <- rp_resolve_attr(default = "GENDER")
GENERATOR <- rp_resolve_generator(default = "CHATGPT4")
scm       <- rp_scm(ATTR)

has_flag <- function(flag) {
  a <- commandArgs(trailingOnly = TRUE)
  any(grepl(paste0("^", flag, "($|=)"), a))
}
PER_SCENARIO <- has_flag("--per-scenario") || has_flag("--per_scenario") ||
  toupper(Sys.getenv("E1_PER_SCENARIO", "")) %in% c("1", "TRUE", "YES")
PLOTS_ONLY <- has_flag("--plots-only") || has_flag("--plots_only") ||
  toupper(Sys.getenv("E1_PLOTS_ONLY", "")) %in% c("1", "TRUE", "YES")

base_sa <- rp_sa_dir(ATTR, GENERATOR)
out_dir <- rp_result_dir(ATTR, GENERATOR, "E1_dag_sensitivity")

log_info(ATTR, " / ", GENERATOR, " | E1 DAG sensitivity of the analysis",
         if (PLOTS_ONLY) "  [plots-only: redraw from saved CSVs]" else "")
log_info("Input (SA results): ", base_sa)
log_info("Output            : ", out_dir)
if (!PLOTS_ONLY && !dir_exists(base_sa)) {
  log_err("Input folder not found: ", base_sa, " (run the sentiment analysis first)")
  quit(status = 1)
}
dir_create(out_dir, recurse = TRUE)

scenarios    <- rp_scenarios(ATTR)
y_candidates <- RP_Y_CANDIDATES

find_scenario_csv <- function(dir_model, s) {
  pat <- paste0("(^|_)s", sub("^s", "", s), ".*\\.csv$",
                "|generated_sentences_", s, ".*\\.csv$",
                "|dataset_with_sentiment_scores_.*_", s, "\\.csv$",
                "|sentiment_.*_", s, ".*\\.csv$")
  p <- tryCatch(fs::dir_ls(dir_model, regexp = pat, type = "file"), error = function(e) character(0))
  if (length(p) > 0) p[1] else character(0)
}

normalize_columns <- function(df) {
  nm <- names(df)
  repl <- c(
    "Employment Status"  = "Employment",
    "employment_status"  = "Employment",
    "employment"         = "Employment",
    "Education Level"     = "Education",
    "education_level"     = "Education",
    "education"           = "Education",
    "Prior Convictions"   = "Prior.Convictions"
  )
  for (k in names(repl)) {
    if (k %in% nm && !(repl[[k]] %in% nm)) names(df)[names(df) == k] <- repl[[k]]
  }
  df
}

pick_ycol <- function(df_list, candidates) {
  cols <- unique(unlist(lapply(df_list, names)))
  y <- intersect(candidates, cols)
  if (length(y)) y[1] else NA_character_
}

short_label <- function(x) {
  xl <- tolower(x)
  dplyr::case_when(
    grepl("^allam[_-]?2[_-]?7b", xl) ~ "allam",
    grepl("^cardiffnlp", xl) ~ "rob_twit",
    grepl("^distilbert", xl) ~ "distil",
    grepl("^gemma2", xl) ~ "gemma",
    grepl("^gpt[_-]?3[_-]?5", xl) ~ "gpt35",
    grepl("^llama[_-]?3[_-]?70b", xl) ~ "llama70b",
    grepl("^llama[_-]?3[_-]?1[_-]?8b", xl) ~ "llama8b",
    grepl("^siebert", xl) ~ "siebert",
    grepl("^textattack", xl) ~ "bert",
    TRUE ~ substr(gsub("[^A-Za-z0-9]", "", x), 1, 10)
  )
}

dag_family <- function(attr, scm) {
  Z0 <- scm$Z
  W0 <- scm$W
  z_txt <- paste(Z0, collapse = ",")
  if (toupper(attr) == "GENDER") {
    w2     <- "Employment"
    w_keep <- setdiff(W0, w2)
    list(
      list(label = "G0", Z = Z0,           W = W0,     name = "Full SFM",
           change = "baseline (full graph)", concern = "-",
           cap = "G0 full SFM", pretty = "G0\nfull SFM"),
      list(label = "G1", Z = Z0,           W = w_keep, name = "Drop mediator",
           change = paste0("remove mediator ", w2),
           concern = "mediator misspecification",
           cap = paste0("G1 drop mediator ", w2),
           pretty = paste0("G1\ndrop ", w2, " (W)")),
      list(label = "G2", Z = character(0), W = W0,     name = "Drop confounder",
           change = paste0("remove confounder ", z_txt),
           concern = "omitted confounder",
           cap = paste0("G2 drop confounder ", z_txt),
           pretty = paste0("G2\ndrop ", z_txt, " (Z)")),
      list(label = "G3", Z = c(Z0, w2),    W = w_keep, name = "Reclassify mediator",
           change = paste0("move ", w2, " from mediator to confounder"),
           concern = "mediator/confounder ambiguity",
           cap = paste0("G3 ", w2, " as confounder"),
           pretty = paste0("G3\n", w2, " as Z"))
    )
  } else {
    list(
      list(label = "G0", Z = Z0,           W = W0, name = "Full SFM",
           change = "baseline (full graph)", concern = "-",
           cap = "G0 full SFM", pretty = "G0\nfull SFM"),
      list(label = "G2", Z = character(0), W = W0, name = "Drop confounder",
           change = paste0("remove confounder ", z_txt),
           concern = "omitted confounder",
           cap = paste0("G2 drop confounder ", z_txt),
           pretty = paste0("G2\ndrop ", z_txt, " (Z)"))
    )
  }
}

run_decomposition_dag <- function(df, ycol, Xcol, Zvars, Wvars, x0, x1, orig_Z) {
  required <- c(Xcol, Zvars, Wvars, ycol)
  missing  <- setdiff(required, names(df))
  if (length(missing)) {
    log_warn("Missing required columns: ", paste(missing, collapse = ", "))
    return(NULL)
  }
  df[[Xcol]] <- as.character(df[[Xcol]])
  for (w in Wvars) df[[w]] <- as.character(df[[w]])
  for (z in Zvars) {
    if (z %in% orig_Z) df[[z]] <- suppressWarnings(as.numeric(df[[z]]))
    else               df[[z]] <- as.character(df[[z]])
  }
  df <- df %>% filter(!is.na(.data[[ycol]]))
  if (!nrow(df)) return(NULL)
  suppressWarnings(
    fairness_cookbook(data = df, X = Xcol, Z = Zvars, W = Wvars, Y = ycol,
                      x0 = x0, x1 = x1, method = "debiasing")
  )
}

COMP_MAP <- c(tv = "TV", ctfde = "DE", ctfie = "IE", ctfse = "SE")
COMP_LEVELS <- c("TV", "DE", "IE", "SE")

extract_measures_ci <- function(res) {
  if (is.null(res) || is.null(res$measures)) return(NULL)
  m <- as.data.frame(res$measures)
  if (!all(c("measure", "value", "sd") %in% names(m))) return(NULL)
  m <- m[m$measure %in% names(COMP_MAP), c("measure", "value", "sd")]
  if (!nrow(m)) return(NULL)
  m$component <- unname(COMP_MAP[m$measure])
  m$ci_lo <- m$value - 1.96 * m$sd
  m$ci_hi <- m$value + 1.96 * m$sd
  m$verdict <- ifelse(m$ci_lo > 0, "+", ifelse(m$ci_hi < 0, "-", "0"))
  tibble::as_tibble(m[, c("component", "value", "sd", "ci_lo", "ci_hi", "verdict")])
}

dags        <- dag_family(ATTR, scm)
dag_labels  <- vapply(dags, `[[`, "", "label")
dag_pretty  <- setNames(vapply(dags, `[[`, "", "pretty"), dag_labels)
dag_name    <- setNames(vapply(dags, `[[`, "", "name"),   dag_labels)
dag_legend  <- setNames(paste0(dag_labels, ": ", dag_name), dag_labels)
log_info("DAG family (", length(dags), "): ", paste(dag_labels, collapse = ", "))
log_info("Scope: pooled (all scenarios)", if (PER_SCENARIO) " + per-scenario" else "")

out_legend  <- fs::path(out_dir, "e1_dag_legend.csv")
out_long    <- fs::path(out_dir, "e1_estimates_long.csv")
out_matrix  <- fs::path(out_dir, "e1_verdict_matrix.csv")
out_summary <- fs::path(out_dir, "e1_robustness_summary.csv")
out_long_ps <- fs::path(out_dir, "e1_estimates_long_per_scenario.csv")

decompose_unit <- function(df_unit, ycol, m, scen_label) {
  rows <- list()
  for (g in dags) {
    res <- tryCatch(
      run_decomposition_dag(df_unit, ycol, scm$X, g$Z, g$W, scm$x0, scm$x1, scm$Z),
      error = function(e) { log_err("[", m, " / ", scen_label, " / ", g$label, "] cookbook error: ", e$message); NULL })
    est <- extract_measures_ci(res)
    if (is.null(est)) {
      log_warn("[", m, " / ", scen_label, " / ", g$label, "] no estimates")
      est <- tibble::tibble(component = COMP_LEVELS, value = NA_real_, sd = NA_real_,
                            ci_lo = NA_real_, ci_hi = NA_real_, verdict = NA_character_)
    }
    est$model    <- m
    est$scenario  <- scen_label
    est$dag       <- g$label
    est$dag_desc  <- g$change
    est$Z_set     <- paste(g$Z, collapse = "+"); if (!length(g$Z)) est$Z_set <- "(none)"
    est$W_set     <- paste(g$W, collapse = "+")
    rows[[length(rows) + 1]] <- est
    log_info("   [", scen_label, " / ", g$label, "] ",
             paste(sprintf("%s=%s", est$component, est$verdict), collapse = " "))
  }
  rows
}

if (PLOTS_ONLY) {
  if (!file.exists(out_long)) {
    log_err("[plots-only] missing ", out_long,
            " - run a full E1 once (without --plots-only) before redrawing.")
    quit(status = 1)
  }
  long <- readr::read_csv(out_long, show_col_types = FALSE) %>%
    mutate(component = factor(component, levels = COMP_LEVELS))
  long_ps <- if (PER_SCENARIO && file.exists(out_long_ps)) {
    readr::read_csv(out_long_ps, show_col_types = FALSE) %>%
      mutate(component = factor(component, levels = COMP_LEVELS))
  } else long[0, ]
  models <- sort(unique(long$model))
  log_info("[plots-only] loaded ", nrow(long), " pooled rows for ", length(models),
           " model(s); per-scenario rows: ", nrow(long_ps))
} else {
models <- tryCatch(fs::dir_ls(base_sa, type = "directory", recurse = FALSE) %>% fs::path_file(),
                   error = function(e) character(0))
if (!length(models)) { log_err("No model folders found under: ", base_sa); quit(status = 1) }
models <- rp_drop_toxbert(models)
if (!length(models)) { log_err("No sentiment models left after excluding toxbert under: ", base_sa); quit(status = 1) }
log_info("Models found (", length(models), "): ", paste(models, collapse = ", "))

long_rows <- list()
for (m in models) {
  in_dir <- fs::path(base_sa, m)
  log_info("== Model: ", m)

  scen_files <- purrr::map(scenarios, ~ find_scenario_csv(in_dir, .x))
  names(scen_files) <- scenarios
  n_found <- sum(lengths(scen_files) > 0)
  if (n_found == 0) { log_warn("[", m, "] no scenario files, skipping"); next }

  dfs <- list()
  for (s in names(scen_files)) {
    f <- scen_files[[s]]
    if (!length(f)) next
    df <- tryCatch(readr::read_csv(f, show_col_types = FALSE),
                   error = function(e) { log_err("[", m, "] read error ", s, ": ", e$message); NULL })
    if (is.null(df)) next
    df <- normalize_columns(df)
    dfs[[s]] <- df
  }
  if (!length(dfs)) { log_warn("[", m, "] all reads failed, skipping"); next }

  ycol <- pick_ycol(dfs, y_candidates)
  if (is.na(ycol)) { log_warn("[", m, "] no valid Y, skipping"); next }
  df_all <- dplyr::bind_rows(lapply(dfs, function(d) d %>% dplyr::filter(!is.na(.data[[ycol]]))))
  log_info("[", m, "] Y=", ycol, " | scenarios=", n_found, " | pooled rows=", nrow(df_all))
  if (!nrow(df_all)) { log_warn("[", m, "] empty pooled data, skipping"); next }

  units <- list(ALL = df_all)
  if (PER_SCENARIO) {
    for (s in names(dfs)) {
      d <- dfs[[s]] %>% dplyr::filter(!is.na(.data[[ycol]]))
      if (nrow(d)) units[[s]] <- d
    }
  }
  for (u in names(units)) {
    long_rows <- c(long_rows, decompose_unit(units[[u]], ycol, m, u))
  }
}

if (!length(long_rows)) { log_err("No estimates produced for any model."); quit(status = 1) }

long_full <- dplyr::bind_rows(long_rows) %>%
  mutate(component = factor(component, levels = COMP_LEVELS)) %>%
  arrange(model, scenario, component, dag)

long    <- long_full %>% filter(scenario == "ALL")
long_ps <- long_full %>% filter(scenario != "ALL")
}

verdict_wide <- long %>%
  select(model, component, dag, verdict) %>%
  tidyr::pivot_wider(names_from = dag, values_from = verdict)

baseline_col <- "G0"
nonbase <- setdiff(dag_labels, baseline_col)
verdict_wide$robust <- apply(verdict_wide, 1, function(r) {
  v0 <- r[[baseline_col]]
  vs <- unlist(r[nonbase])
  if (is.na(v0) || any(is.na(vs))) return(NA)
  all(vs == v0)
})
verdict_wide <- verdict_wide %>% arrange(component, model)

cell <- long %>%
  filter(dag != baseline_col) %>%
  select(model, component, dag, verdict) %>%
  left_join(long %>% filter(dag == baseline_col) %>% select(model, component, base = verdict),
            by = c("model", "component")) %>%
  mutate(match = !is.na(verdict) & !is.na(base) & verdict == base)

rob_by_comp <- cell %>%
  group_by(component) %>%
  summarise(n_cells = n(), n_robust = sum(match), .groups = "drop") %>%
  mutate(robust_pct = round(100 * n_robust / pmax(n_cells, 1), 1))
rob_all <- tibble::tibble(component = "ALL",
                          n_cells = sum(rob_by_comp$n_cells),
                          n_robust = sum(rob_by_comp$n_robust)) %>%
  mutate(robust_pct = round(100 * n_robust / pmax(n_cells, 1), 1))
rob_summary <- bind_rows(rob_by_comp %>% mutate(component = as.character(component)), rob_all)

legend_tbl <- tibble::tibble(
  dag              = dag_labels,
  name             = vapply(dags, `[[`, "", "name"),
  Z_confounders    = vapply(dags, function(g) if (length(g$Z)) paste(g$Z, collapse = "+") else "(none)", ""),
  W_mediators      = vapply(dags, function(g) if (length(g$W)) paste(g$W, collapse = "+") else "(none)", ""),
  change_vs_G0     = vapply(dags, `[[`, "", "change"),
  reviewer_concern = vapply(dags, `[[`, "", "concern")
)
readr::write_csv(legend_tbl, out_legend)
if (!PLOTS_ONLY) readr::write_csv(long %>% mutate(component = as.character(component)), out_long)
readr::write_csv(verdict_wide %>% mutate(component = as.character(component)), out_matrix)
readr::write_csv(rob_summary, out_summary)

rob_ps  <- NULL
have_ps <- PER_SCENARIO && nrow(long_ps) > 0
if (have_ps) {
  if (!PLOTS_ONLY) readr::write_csv(long_ps %>% mutate(component = as.character(component)), out_long_ps)

  ps_cell <- long_ps %>%
    filter(dag != baseline_col) %>%
    select(model, scenario, component, dag, verdict) %>%
    left_join(long_ps %>% filter(dag == baseline_col) %>%
                select(model, scenario, component, base = verdict),
              by = c("model", "scenario", "component")) %>%
    mutate(match = !is.na(verdict) & !is.na(base) & verdict == base)
  rob_ps <- ps_cell %>%
    group_by(scenario, component) %>%
    summarise(n_cells = n(), n_robust = sum(match), .groups = "drop") %>%
    mutate(robust_pct = round(100 * n_robust / pmax(n_cells, 1), 1))
  if (!PLOTS_ONLY) readr::write_csv(rob_ps %>% mutate(component = as.character(component)),
                   fs::path(out_dir, "e1_robustness_per_scenario.csv"))
}

plot_files <- character(0)
save_plot <- function(p, fname, width, height) {
  f <- fs::path(out_dir, fname)
  ggsave(f, p, width = width, height = height, device = grDevices::cairo_pdf)
  plot_files <<- c(plot_files, as.character(f))
  invisible(f)
}
n_models  <- length(models)
paper_theme <- theme_minimal(base_size = 16) +
  theme(legend.position = "bottom", legend.title = element_text(size = 14),
        legend.text = element_text(size = 13),
        strip.text = element_text(face = "bold"),
        panel.grid.minor = element_blank())

dag_cap <- paste(vapply(dags, function(g) paste0(g$label, ": ", g$change), ""),
                 collapse = "    |    ")

pdat <- long %>% filter(is.finite(value))
if (nrow(pdat)) {
  pdat$model_short <- vapply(as.character(pdat$model), short_label, character(1))
  pdat$dag <- factor(pdat$dag, levels = dag_labels)

  tryCatch({
    p1 <- ggplot(pdat, aes(x = dag, y = value, color = model_short, group = model_short)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      geom_point(position = position_dodge(width = 0.6), size = 3) +
      geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                    position = position_dodge(width = 0.6), width = 0.3) +
      facet_wrap(~component, nrow = 1, scales = "free_y") +
      labs(x = "Analysis graph", y = "Estimate (95% CI)", color = "Model",
           caption = dag_cap) +
      paper_theme + theme(legend.title = element_blank(),
                          plot.caption = element_text(hjust = 0, size = 12))
    save_plot(p1, "e1_dag_estimates.pdf", 15, 7)
  }, error = function(e) log_warn("plot e1_dag_estimates failed: ", e$message))

  tryCatch({
    p2 <- ggplot(pdat, aes(x = component, y = value, color = dag, group = dag)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      geom_point(position = position_dodge(width = 0.6), size = 3) +
      geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                    position = position_dodge(width = 0.6), width = 0.3) +
      facet_wrap(~model_short, scales = "free_y", ncol = min(3, n_models)) +
      scale_color_discrete(labels = dag_legend) +
      labs(x = "Decomposition component", y = "Estimate (95% CI)",
           color = "Analysis graph") +
      paper_theme + guides(color = guide_legend(nrow = 1))
    save_plot(p2, "e1_dag_estimates_by_model.pdf",
              3.3 * min(3, n_models) + 1.2, 2.9 * ceiling(n_models / min(3, n_models)) + 1.6)
  }, error = function(e) log_warn("plot e1_dag_estimates_by_model failed: ", e$message))

  tryCatch({
    base_vals <- pdat %>% filter(dag == "G0") %>% select(model, component, base_value = value)
    ddat <- pdat %>% filter(dag != "G0", component != "TV") %>%
      left_join(base_vals, by = c("model", "component")) %>%
      mutate(delta = value - base_value) %>% filter(is.finite(delta))
    if (nrow(ddat)) {
      ddat$component <- droplevels(factor(ddat$component, levels = COMP_LEVELS))
      ddat$dag       <- droplevels(factor(ddat$dag, levels = dag_labels))
      nonbase_dags <- Filter(function(g) g$label != "G0", dags)
      delta_cap <- paste(vapply(nonbase_dags, function(g) paste0(g$label, ": ", g$change), ""),
                         collapse = "    |    ")
      p3 <- ggplot(ddat, aes(x = dag, y = delta, fill = model_short)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
        geom_col(position = position_dodge(width = 0.8), width = 0.7) +
        facet_wrap(~component, nrow = 1, scales = "free_y") +
        labs(x = "Analysis graph (vs. baseline G0)",
             y = "Change in estimate vs. G0", fill = "Model",
             caption = delta_cap) +
        paper_theme + theme(legend.title = element_blank(),
                            plot.caption = element_text(hjust = 0, size = 12))
      save_plot(p3, "e1_delta_from_baseline.pdf", 15, 7)
    }
  }, error = function(e) log_warn("plot e1_delta_from_baseline failed: ", e$message))
}

tryCatch({
  rdat <- rob_summary %>%
    mutate(component = factor(component, levels = c(COMP_LEVELS, "ALL")))
  p5 <- ggplot(rdat, aes(x = component, y = robust_pct, fill = component)) +
    geom_col(width = 0.7, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.0f%%", robust_pct)), vjust = -0.3, size = 5.5) +
    coord_cartesian(ylim = c(0, 108)) +
    labs(x = "Decomposition component", y = "Conclusions matching baseline G0 (%)") +
    paper_theme + theme(legend.position = "none")
  save_plot(p5, "e1_robustness_bars.pdf", 8.5, 5.6)
}, error = function(e) log_warn("plot e1_robustness_bars failed: ", e$message))

if (PER_SCENARIO && nrow(long_ps)) {
  psdat <- long_ps %>% filter(is.finite(value))
  if (nrow(psdat)) {
    psdat$model_short <- vapply(as.character(psdat$model), short_label, character(1))
    psdat$dag <- factor(psdat$dag, levels = dag_labels)
    psdat$scenario <- factor(psdat$scenario,
                             levels = scenarios[scenarios %in% unique(psdat$scenario)])

    tryCatch({
      de <- psdat %>% filter(component == "DE")
      if (nrow(de)) {
        p6 <- ggplot(de, aes(x = scenario, y = value, color = model_short, group = model_short)) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
          geom_point(size = 2.4) + geom_line(linewidth = 0.7) +
          facet_wrap(~dag, nrow = 1, labeller = labeller(dag = dag_legend)) +
          labs(x = "Scenario", y = "Direct effect (DE)", color = "Model") +
          paper_theme +
          theme(legend.title = element_blank(),
                axis.text.x = element_text(angle = 45, hjust = 1))
        save_plot(p6, "e1_per_scenario_DE.pdf", 15, 6.5)
      }
    }, error = function(e) log_warn("plot e1_per_scenario_DE failed: ", e$message))

    tryCatch({
      p7 <- ggplot(psdat, aes(x = scenario, y = value, color = model_short, group = model_short)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
        geom_point(size = 1.7) + geom_line(linewidth = 0.6) +
        facet_grid(component ~ dag, scales = "free_y", labeller = labeller(dag = dag_legend)) +
        labs(x = "Scenario", y = "Estimate", color = "Model") +
        paper_theme +
        theme(legend.title = element_blank(),
              axis.text.x = element_text(angle = 45, hjust = 1, size = 9))
      save_plot(p7, "e1_per_scenario_estimates.pdf", 15, 11)
    }, error = function(e) log_warn("plot e1_per_scenario_estimates failed: ", e$message))
  }

}

plotted <- length(plot_files) > 0

cat(strrep("=", 70), "\n", sep = "")
cat("E1 - DAG sensitivity of the analysis | ", ATTR, " / ", GENERATOR, "\n", sep = "")
cat(strrep("=", 70), "\n", sep = "")
cat("DAG family:\n")
for (g in dags) cat(sprintf("  %-3s  Z=(%s)  W=(%s)   %s\n",
                            g$label,
                            if (length(g$Z)) paste(g$Z, collapse = ",") else "",
                            paste(g$W, collapse = ","), g$change))
cat("\nVerdict matrix (v in {+, -, 0}; robust = matches G0 on every graph):\n")
print(as.data.frame(verdict_wide), row.names = FALSE)
cat("\nRobustness (share of model x graph conclusions agreeing with G0):\n")
print(as.data.frame(rob_summary), row.names = FALSE)
cat(sprintf("\n[ok] Overall: %s%% of (model x graph x component) conclusions are graph-robust.\n",
            rob_all$robust_pct))
cat("     (TV is graph-invariant and should read 100%; DE/IE/SE test the decomposition.)\n")
cat(sprintf("\nScope: pooled (all scenarios)%s\n",
            if (PER_SCENARIO) " + per-scenario" else " [add --per-scenario for per-scenario analysis]"))
cat(if (PLOTS_ONLY) "\nWrote (plots-only: figures + derived tables; data CSVs left intact):\n" else "\nWrote:\n")
cat("  - ", out_legend, "\n", sep = "")
if (!PLOTS_ONLY) cat("  - ", out_long, "\n", sep = "")
cat("  - ", out_matrix, "\n", sep = "")
cat("  - ", out_summary, "\n", sep = "")
if (have_ps && !PLOTS_ONLY) cat("  - ", out_long_ps, " (per-scenario)\n", sep = "")
if (isTRUE(plotted)) for (f in plot_files) cat("  - ", f, "\n", sep = "")

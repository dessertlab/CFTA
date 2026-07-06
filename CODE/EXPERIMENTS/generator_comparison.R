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

ATTR <- rp_resolve_attr(default = "GENDER")
scm  <- rp_scm(ATTR)
GENS <- RP_GENERATORS
gen_ini <- c(CHATGPT4 = "C", GEMINIPRO = "G", TEMPLATE = "T")
gen_col <- c(CHATGPT4 = "#1b9e77", GEMINIPRO = "#7570b3", TEMPLATE = "#d95f02")

scenarios    <- rp_scenarios(ATTR)
y_candidates <- RP_Y_CANDIDATES

gc_root <- file.path(rp_result_root(), toupper(ATTR), "GENERATOR_COMPARISON")
agg_dir <- file.path(gc_root, "AGGREGATE")
ps_dir  <- file.path(gc_root, "PER_SCENARIO")
dir_create(agg_dir, recurse = TRUE)
dir_create(ps_dir,  recurse = TRUE)

has_flag <- function(flag) {
  a <- commandArgs(trailingOnly = TRUE)
  any(grepl(paste0("^", flag, "($|=)"), a))
}
PLOTS_ONLY <- has_flag("--plots-only") || has_flag("--plots_only") ||
  toupper(Sys.getenv("GC_PLOTS_ONLY", "")) %in% c("1", "TRUE", "YES")

agg_long_csv <- file.path(agg_dir, "gen_comparison_long.csv")
ps_long_csv  <- file.path(ps_dir,  "gen_comparison_long_per_scenario.csv")

fac_comp <- function(d) mutate(d, component = factor(component, levels = COMP_LEVELS))
fac_gen  <- function(d) mutate(d, generator = factor(generator, levels = GENS))

log_info(ATTR, " | generator comparison (SFM decomposition across generators)",
         if (PLOTS_ONLY) "  [plots-only: redraw from saved CSVs]" else "")
log_info("Output: ", gc_root)

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
  repl <- c("Employment Status" = "Employment", "employment_status" = "Employment",
            "employment" = "Employment", "Education Level" = "Education",
            "education_level" = "Education", "education" = "Education",
            "Prior Convictions" = "Prior.Convictions")
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

COMP_MAP    <- c(tv = "TV", ctfde = "DE", ctfie = "IE", ctfse = "SE")
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

decompose_sfm <- function(df, ycol) {
  required <- c(scm$X, scm$Z, scm$W, ycol)
  if (length(setdiff(required, names(df)))) return(NULL)
  df[[scm$X]] <- as.character(df[[scm$X]])
  for (w in scm$W) df[[w]] <- as.character(df[[w]])
  for (z in scm$Z) df[[z]] <- suppressWarnings(as.numeric(df[[z]]))
  df <- df %>% filter(!is.na(.data[[ycol]]))
  if (!nrow(df)) return(NULL)
  res <- tryCatch(
    suppressWarnings(fairness_cookbook(data = df, X = scm$X, Z = scm$Z, W = scm$W,
                                       Y = ycol, x0 = scm$x0, x1 = scm$x1, method = "debiasing")),
    error = function(e) { log_err("cookbook error: ", e$message); NULL })
  extract_measures_ci(res)
}

if (PLOTS_ONLY) {
  if (!file.exists(agg_long_csv)) {
    log_err("[plots-only] missing ", agg_long_csv,
            " - run a full generator comparison once (without --plots-only) before redrawing.")
    quit(status = 1)
  }
  pooled <- readr::read_csv(agg_long_csv, show_col_types = FALSE) %>%
    fac_comp() %>% fac_gen() %>% mutate(model_short = vapply(model, short_label, character(1)))
  ps <- if (file.exists(ps_long_csv)) {
    readr::read_csv(ps_long_csv, show_col_types = FALSE) %>%
      fac_comp() %>% fac_gen() %>% mutate(model_short = vapply(model, short_label, character(1)))
  } else NULL
  present <- intersect(GENS, unique(as.character(pooled$generator)))
  common  <- sort(unique(pooled$model))
  log_info("[plots-only] loaded pooled rows=", nrow(pooled),
           " | generators: ", paste(present, collapse = ", "),
           " | models (", length(common), "): ", paste(common, collapse = ", "))
} else {
present <- GENS[vapply(GENS, function(g) dir_exists(rp_sa_dir(ATTR, g)), logical(1))]
if (length(present) < 2) {
  log_err("Need at least 2 scored generators; found: ", paste(present, collapse = ", "))
  quit(status = 1)
}
models_by_gen <- lapply(present, function(g) {
  ms <- tryCatch(fs::dir_ls(rp_sa_dir(ATTR, g), type = "directory") %>% fs::path_file(),
                 error = function(e) character(0))
  rp_drop_toxbert(ms)
})
names(models_by_gen) <- present
common <- Reduce(intersect, models_by_gen)
if (!length(common)) {
  log_err("No SA model is present across all generators ", paste(present, collapse = ", "))
  quit(status = 1)
}
log_info("Generators: ", paste(present, collapse = ", "))
log_info("Common models (", length(common), "): ", paste(common, collapse = ", "))

pooled_rows <- list()
ps_rows     <- list()
for (g in present) {
  for (m in common) {
    in_dir <- fs::path(rp_sa_dir(ATTR, g), m)
    scen_files <- purrr::map(scenarios, ~ find_scenario_csv(in_dir, .x))
    names(scen_files) <- scenarios
    dfs <- list()
    for (s in names(scen_files)) {
      f <- scen_files[[s]]
      if (!length(f)) next
      df <- tryCatch(readr::read_csv(f, show_col_types = FALSE), error = function(e) NULL)
      if (!is.null(df)) dfs[[s]] <- normalize_columns(df)
    }
    if (!length(dfs)) { log_warn("[", g, "/", m, "] no scenario files"); next }
    ycol <- pick_ycol(dfs, y_candidates)
    if (is.na(ycol)) { log_warn("[", g, "/", m, "] no valid Y"); next }

    df_all <- dplyr::bind_rows(lapply(dfs, function(d) d %>% filter(!is.na(.data[[ycol]]))))
    est <- decompose_sfm(df_all, ycol)
    if (!is.null(est)) {
      est$generator <- g; est$model <- m
      pooled_rows[[length(pooled_rows) + 1]] <- est
      log_info("[", g, "/", m, "] pooled Y=", ycol, " rows=", nrow(df_all), " ",
               paste(sprintf("%s=%s", est$component, est$verdict), collapse = " "))
    }
    for (s in names(dfs)) {
      d <- dfs[[s]] %>% filter(!is.na(.data[[ycol]]))
      if (!nrow(d)) next
      e <- decompose_sfm(d, ycol)
      if (is.null(e)) next
      e$generator <- g; e$model <- m; e$scenario <- s
      ps_rows[[length(ps_rows) + 1]] <- e
    }
  }
}
if (!length(pooled_rows)) { log_err("No pooled estimates produced."); quit(status = 1) }

pooled <- dplyr::bind_rows(pooled_rows) %>% fac_comp() %>% fac_gen() %>%
  mutate(model_short = vapply(model, short_label, character(1)))
ps     <- if (length(ps_rows)) dplyr::bind_rows(ps_rows) %>% fac_comp() %>% fac_gen() %>%
  mutate(model_short = vapply(model, short_label, character(1))) else NULL
}

concordance <- function(d, keys) {
  d %>% arrange(across(all_of(c(keys, "generator")))) %>%
    group_by(across(all_of(keys))) %>%
    summarise(
      n_gen  = sum(!is.na(verdict)),
      agreed = { v <- unique(verdict[!is.na(verdict)]); if (length(v) == 1) v else NA_character_ },
      verdicts = paste(sprintf("%s:%s", gen_ini[as.character(generator)],
                               ifelse(is.na(verdict), "?", verdict)), collapse = " "),
      .groups = "drop") %>%
    mutate(concordant = !is.na(agreed) & n_gen >= 2,
           status = dplyr::case_when(!is.na(agreed) ~ agreed, n_gen >= 2 ~ "mixed", TRUE ~ "n/a"))
}

to_wide <- function(d, keys, conc) {
  w <- d %>% select(all_of(c(keys, "generator")), value, ci_lo, ci_hi, verdict) %>%
    pivot_wider(names_from = generator, values_from = c(value, ci_lo, ci_hi, verdict),
                names_glue = "{generator}_{.value}")
  w %>% left_join(conc %>% select(all_of(keys), verdict_concordant = concordant,
                                  verdicts, agreed_verdict = agreed),
                  by = keys)
}

conc_agg <- concordance(pooled, c("model", "component"))
agg_wide <- to_wide(pooled, c("model", "component"), conc_agg)
if (!PLOTS_ONLY) readr::write_csv(pooled %>% mutate(component = as.character(component)),
                 file.path(agg_dir, "gen_comparison_long.csv"))
readr::write_csv(agg_wide %>% mutate(component = as.character(component)),
                 file.path(agg_dir, "gen_comparison_estimates.csv"))

conc_ps <- NULL
if (!is.null(ps)) {
  conc_ps <- concordance(ps, c("model", "scenario", "component"))
  ps_wide <- to_wide(ps, c("model", "scenario", "component"), conc_ps)
  if (!PLOTS_ONLY) readr::write_csv(ps %>% mutate(component = as.character(component)),
                   file.path(ps_dir, "gen_comparison_long_per_scenario.csv"))
  readr::write_csv(ps_wide %>% mutate(component = as.character(component)),
                   file.path(ps_dir, "gen_comparison_estimates_per_scenario.csv"))
}

plot_files <- character(0)
save_plot <- function(p, path, width, height) {
  ggsave(path, p, width = width, height = height, device = grDevices::cairo_pdf)
  plot_files <<- c(plot_files, as.character(path)); invisible(path)
}
status_cols <- c("+" = "#1a9850", "-" = "#d73027", "0" = "#f7f7f7",
                 "mixed" = "#fdae61", "n/a" = "grey85")
n_models  <- length(common)
paper_theme <- theme_minimal(base_size = 16) +
  theme(legend.position = "bottom", legend.title = element_text(size = 14),
        legend.text = element_text(size = 13),
        strip.text = element_text(face = "bold"),
        panel.grid.minor = element_blank())

tryCatch({
  pa <- pooled %>% filter(is.finite(value))
  ncol_a  <- min(3, n_models)
  nrow_a  <- ceiling(n_models / ncol_a)
  p <- ggplot(pa, aes(x = component, y = value, color = generator, group = generator)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_point(position = position_dodge(width = 0.6), size = 3) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                  position = position_dodge(width = 0.6), width = 0.3) +
    facet_wrap(~model_short, scales = "free_y", ncol = ncol_a) +
    scale_color_manual(values = gen_col, name = "Generator") +
    labs(x = "Decomposition component", y = "Estimate (95% CI)") +
    paper_theme
  save_plot(p, file.path(agg_dir, "gen_comparison_estimates.pdf"),
            3.3 * ncol_a + 1.2, 2.9 * nrow_a + 1.4)
}, error = function(e) log_warn("plot aggregate estimates failed: ", e$message))

if (!is.null(ps)) {
  tryCatch({
    de <- ps %>% filter(component == "DE", is.finite(value)) %>%
      mutate(scenario = factor(scenario, levels = scenarios[scenarios %in% unique(scenario)]))
    p <- ggplot(de, aes(x = scenario, y = value, color = generator, group = generator)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      geom_point(position = position_dodge(width = 0.5), size = 2.4) +
      geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                    position = position_dodge(width = 0.5), width = 0.3) +
      facet_wrap(~model_short, ncol = 1, scales = "free_y") +
      scale_color_manual(values = gen_col, name = "Generator") +
      labs(x = "Scenario", y = "Direct effect (DE), 95% CI") +
      paper_theme +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    save_plot(p, file.path(ps_dir, "gen_comparison_DE_per_scenario.pdf"),
              15, 2.4 * n_models + 2)
  }, error = function(e) log_warn("plot per-scenario DE failed: ", e$message))

}

cat(strrep("=", 70), "\n", sep = "")
cat("Generator comparison | ", ATTR, "  (", paste(present, collapse = ", "), ")\n", sep = "")
cat(strrep("=", 70), "\n", sep = "")
cat("Common models: ", paste(common, collapse = ", "), "\n\n", sep = "")
cat("Aggregate verdict concordance (per model x component):\n")
print(as.data.frame(conc_agg %>% select(model, component, verdicts, concordant)), row.names = FALSE)
cat(sprintf("\n[ok] Aggregate cells where all generators agree on the sign: %d/%d\n",
            sum(conc_agg$concordant), nrow(conc_agg)))
cat("\nWrote:\n")
for (f in plot_files) cat("  - ", f, "\n", sep = "")
cat("  - ", file.path(agg_dir, "gen_comparison_estimates.csv"), "\n", sep = "")
if (!is.null(ps)) cat("  - ", file.path(ps_dir, "gen_comparison_estimates_per_scenario.csv"), "\n", sep = "")

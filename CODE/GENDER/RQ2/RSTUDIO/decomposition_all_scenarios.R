suppressWarnings({
  library(readr)
  library(fs)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(ggplot2)
  library(faircause)
})

options(warn = 1)
set.seed(42)

now_ts  <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
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

base_sa  <- rp_sa_dir(ATTR, GENERATOR)
out_root <- rp_rstudio_all_dir(ATTR, GENERATOR)
log_info(ATTR, " / ", GENERATOR, " | pooled (all-scenarios) decomposition")
log_info("Expecting input under: ", base_sa)
log_info("Output will go to:     ", out_root)

if (!fs::dir_exists(base_sa)) {
  log_err("Input folder not found: ", base_sa, " (run the sentiment analysis first)")
  quit(status = 1)
}
fs::dir_create(out_root, recurse = TRUE)

scenarios    <- rp_scenarios(ATTR)
y_candidates <- RP_Y_CANDIDATES

find_scenario_csv <- function(dir_model, s) {
  pat <- paste0("(^|_)s", sub("^s","",s), ".*\\.csv$",
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
    "Education Level"    = "Education",
    "education_level"    = "Education",
    "education"          = "Education",
    "Prior Convictions"  = "Prior.Convictions"
  )
  for (k in names(repl)) {
    if (k %in% nm && !(repl[[k]] %in% nm)) {
      names(df)[names(df) == k] <- repl[[k]]
    }
  }
  df
}

pick_ycol <- function(df_list, candidates) {
  cols <- unique(unlist(lapply(df_list, names)))
  y <- intersect(candidates, cols)
  if (length(y)) y[1] else NA_character_
}

run_decomposition <- function(df, ycol) {
  required <- c(scm$X, scm$Z, scm$W, ycol)
  missing  <- setdiff(required, names(df))
  if (length(missing)) {
    log_warn("Missing required columns: ", paste(missing, collapse=", "))
    return(NULL)
  }

  df[[scm$X]] <- as.character(df[[scm$X]])
  for (z in scm$Z) df[[z]] <- suppressWarnings(as.numeric(df[[z]]))
  for (w in scm$W) df[[w]] <- as.character(df[[w]])
  df <- df %>% filter(!is.na(.data[[ycol]]))

  if (!nrow(df)) return(NULL)

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

models <- tryCatch(fs::dir_ls(base_sa, type = "directory", recurse = FALSE) %>% fs::path_file(),
                   error = function(e) character(0))
models <- rp_drop_toxbert(models)
if (!length(models)) {
  log_err("No model folders found under: ", base_sa)
  quit(status = 1)
} else {
  log_info("Models found (", length(models), "): ", paste(models, collapse = ", "))
}

for (m in models) {
  in_dir  <- fs::path(base_sa, m)
  out_dir <- fs::path(out_root, m)
  fs::dir_create(out_dir, recurse = TRUE)

  log_info("== Start model: ", m)
  scen_files <- purrr::map(scenarios, ~ find_scenario_csv(in_dir, .x))
  names(scen_files) <- scenarios

  n_found <- sum(lengths(scen_files) > 0)
  log_info("[", m, "] scenarios expected: ", length(scenarios), " | found: ", n_found)
  if (n_found == 0) { log_warn("[", m, "] no scenario files found, skipping"); next }

  dfs <- list()
  for (s in names(scen_files)) {
    f <- scen_files[[s]]
    if (!length(f)) { log_warn("[", m, "] missing file for ", s); next }
    log_info("[", m, "] reading ", s, " -> ", f)
    df <- tryCatch(readr::read_csv(f, show_col_types = FALSE),
                   error = function(e) { log_err("[", m, "] read error: ", s, " | ", e$message); NULL })
    if (is.null(df)) next
    df <- normalize_columns(df)
    df$.__scenario__ <- s
    dfs[[s]] <- df
  }

  if (!length(dfs)) { log_warn("[", m, "] all scenario reads failed, skipping"); next }

  ycol <- pick_ycol(dfs, y_candidates)
  if (is.na(ycol)) { log_warn("[", m, "] no valid Y among candidates, skipping"); next }
  log_info("[", m, "] chosen Y: ", ycol)

  used_scenarios <- 0
  dfs2 <- lapply(dfs, function(d) {
    total <- nrow(d)
    d2    <- d %>% dplyr::filter(!is.na(.data[[ycol]]))
    used  <- nrow(d2)
    sc    <- unique(d$.__scenario__)
    log_info("[", m, "] ", sc, " rows: total=", total, " | used=", used)
    if (used > 0) used_scenarios <<- used_scenarios + 1
    d2
  })

  df_all <- dplyr::bind_rows(dfs2)
  log_info("[", m, "] scenarios used: ", used_scenarios, " | combined rows: ", nrow(df_all))
  if (!nrow(df_all)) { log_warn("[", m, "] empty combined dataset after NA drop, skipping"); next }

  res <- tryCatch(run_decomposition(df_all, ycol),
                  error = function(e) { log_err("[", m, "] fairness_cookbook error: ", e$message); NULL })
  if (is.null(res)) { log_warn("[", m, "] decomposition failed, skipping"); next }

  pdf_path <- fs::path(out_dir, paste0(m, "_all_scenarios_decomposition.pdf"))
  log_info("[", m, "] saving PDF -> ", pdf_path)
  tryCatch({
    pdf(pdf_path, width = 7, height = 5)
    print(autoplot(res))
    dev.off()
  }, error = function(e) {
    log_err("[", m, "] error while saving PDF: ", e$message)
    try(dev.off(), silent = TRUE)
  })

  log_info("== End model: ", m)
}

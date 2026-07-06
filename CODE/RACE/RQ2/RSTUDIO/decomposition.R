suppressWarnings({
  library(readr)
  library(fs)
  library(faircause)
  library(ggplot2)
})

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

base_sa   <- rp_sa_dir(ATTR, GENERATOR)
scenarios <- rp_scenarios(ATTR)
y_candidates <- RP_Y_CANDIDATES

cat(sprintf("Decomposition (per scenario) | %s / %s\n", ATTR, GENERATOR))
cat(sprintf("  input : %s\n", base_sa))
if (!dir_exists(base_sa)) {
  stop(sprintf("Input folder not found: %s (run the sentiment analysis first)", base_sa))
}
models <- rp_drop_toxbert(dir_ls(base_sa, type = "directory", recurse = FALSE))
cat(sprintf("  models: %d found\n", length(models)))

find_scenario_csv <- function(dir_model, s) {
  pat <- paste0(
    "(^|_)s", sub("^s","",s), ".*\\.csv$",
    "|generated_sentences_", s, ".*\\.csv$",
    "|dataset_with_sentiment_scores_.*_", s, "\\.csv$"
  )
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

for (m_dir in models) {
  model_name <- path_file(m_dir)
  out_dir <- path(rp_rstudio_dir(ATTR, GENERATOR), model_name)
  dir_create(out_dir, recurse = TRUE)

  for (s in scenarios) {
    f <- find_scenario_csv(m_dir, s)
    if (length(f) == 0) next

    df <- tryCatch(readr::read_csv(f, show_col_types = FALSE), error = function(e) NULL)
    if (is.null(df)) next

    for (ycol in y_candidates) {
      if (!ycol %in% names(df)) next

      res <- tryCatch(run_decomposition(df, ycol), error = function(e) NULL)
      if (is.null(res)) next

      pdf_path  <- path(out_dir, paste0(s, "_", model_name, "_decomposition.pdf"))
      pdf(pdf_path, width = 7, height = 5)
      print(autoplot(res))
      dev.off()

      break
    }
  }
}

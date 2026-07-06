RP_ATTRS        <- c("GENDER", "RACE")
RP_GENERATORS   <- c("CHATGPT4", "GEMINIPRO", "TEMPLATE")
RP_N_SCENARIOS  <- list(GENDER = 14L, RACE = 12L)

rp_init <- function(root) {
  options(REPL_REPO_ROOT = normalizePath(root, winslash = "/", mustWork = FALSE))
  invisible(getOption("REPL_REPO_ROOT"))
}

rp_root <- function() {
  r <- getOption("REPL_REPO_ROOT", NULL)
  if (is.null(r)) stop("Repo root not set. Call rp_init(root) first (see bootstrap in r_paths.R).")
  r
}

rp_cli_arg <- function(...) {
  flags <- c(...)
  a <- commandArgs(trailingOnly = TRUE)
  for (flag in flags) {
    hit <- grep(paste0("^", flag, "($|=)"), a)
    if (length(hit)) {
      tok <- a[hit[1]]
      if (grepl("=", tok, fixed = TRUE)) return(sub(paste0("^", flag, "="), "", tok))
      if (hit[1] < length(a)) return(a[hit[1] + 1])
    }
  }
  NULL
}

rp_resolve <- function(value, env_var, choices, default = NULL, name = "value") {
  raw <- if (!is.null(value) && nzchar(value)) value else NA_character_
  if (is.na(raw)) {
    e <- Sys.getenv(env_var, unset = NA_character_)
    if (!is.na(e) && nzchar(e)) raw <- e
  }
  if (is.na(raw)) raw <- default
  if (is.null(raw) || is.na(raw)) {
    stop(sprintf("Missing --%s (or $%s). Choose one of: %s",
                 tolower(name), env_var, paste(choices, collapse = ", ")))
  }
  norm <- toupper(raw)
  if (!norm %in% choices) {
    stop(sprintf("Invalid %s '%s'. Choose one of: %s",
                 name, raw, paste(choices, collapse = ", ")))
  }
  norm
}

rp_resolve_attr <- function(default = NULL) {
  rp_resolve(rp_cli_arg("--attr", "-a"), "ATTR", RP_ATTRS, default = default, name = "attr")
}

rp_resolve_generator <- function(default = "CHATGPT4") {
  rp_resolve(rp_cli_arg("--generator", "-g"), "GENERATOR", RP_GENERATORS,
             default = default, name = "generator")
}

rp_scenarios <- function(attr) {
  sprintf("s%d", seq_len(RP_N_SCENARIOS[[toupper(attr)]]))
}

rp_dataset_root <- function() file.path(rp_root(), "DATASET")

rp_synthetic_dir <- function(attr) {
  file.path(rp_dataset_root(), toupper(attr), "SYNTHETIC_DATA")
}
rp_synthetic_csv <- function(attr, s) {
  file.path(rp_synthetic_dir(attr), paste0("synthetic_data_", s, ".csv"))
}
rp_generated_dir <- function(attr, generator) {
  file.path(rp_dataset_root(), toupper(attr), toupper(generator), "GENERATED_SENTENCES")
}
rp_generated_csv <- function(attr, generator, s) {
  file.path(rp_generated_dir(attr, generator), paste0("generated_sentences_", s, ".csv"))
}
rp_complete_csv <- function(attr, generator) {
  file.path(rp_dataset_root(), toupper(attr), toupper(generator), "COMPLETE_DATASET",
            paste0("generated_sentences_", tolower(attr), "_complete.csv"))
}

rp_result_root <- function() file.path(rp_root(), "RESULT")

rp_result_dir <- function(attr, generator, ...) {
  parts <- c(...)
  base <- file.path(rp_result_root(), toupper(attr), toupper(generator))
  if (length(parts)) do.call(file.path, as.list(c(base, parts))) else base
}

rp_rq1_dir          <- function(attr, generator) rp_result_dir(attr, generator, "RQ1")
rp_sa_dir           <- function(attr, generator) rp_result_dir(attr, generator, "RQ2", "SA")
rp_rstudio_dir      <- function(attr, generator) rp_result_dir(attr, generator, "RQ2", "RSTUDIO")
rp_rstudio_all_dir  <- function(attr, generator) rp_result_dir(attr, generator, "RQ2", "RSTUDIO_ALL_SCENARIOS")
rp_rq3_dir          <- function(attr, generator) rp_result_dir(attr, generator, "RQ3")

rp_scm <- function(attr) {
  switch(toupper(attr),
    GENDER = list(X = "Gender", Z = "Age", W = c("Employment", "Education"),
                  x0 = "Female", x1 = "Male"),
    RACE   = list(X = "Race", Z = "Age", W = c("Prior.Convictions"),
                  x0 = "White-Caucasian", x1 = "Non-White"),
    stop(sprintf("Unknown attribute '%s'", attr))
  )
}

RP_Y_CANDIDATES <- c("sentiment_score", "sentiment_score_positive",
                     "sentiment_score_positivo", "toxicity_score")

rp_drop_toxbert <- function(models) models[!grepl("^unitary[_-]?toxic", tolower(basename(models)))]

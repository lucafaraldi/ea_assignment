#!/usr/bin/env Rscript

# Minimal IOHanalyzer usage - direct sourcing from GitHub repo
# This uses the IOHanalyzer code directly without full package installation

# Source the necessary R files from IOHanalyzer
source('/tmp/IOHanalyzer/R/readFiles.R')
source('/tmp/IOHanalyzer/R/DataSet.R')

# Load data from IOHprofiler format
data_dir <- 'ioh_data'

tryCatch({
  # Read IOH data files
  cat("Attempting to load IOH data from:", data_dir, "\n")

  # List available data files
  json_files <- list.files(data_dir, pattern = "*.json$", full.names = TRUE, recursive = TRUE)
  cat("Found JSON files:\n")
  print(json_files)

  # Try basic data reading
  if (length(json_files) > 0) {
    cat("\nAttempting to read first file:", json_files[1], "\n")
    # This will likely fail without all dependencies, but let's try
    result <- try(read_json_file(json_files[1]))
    print(result)
  }

}, error = function(e) {
  cat("Error:", conditionMessage(e), "\n")
  cat("\nIOHanalyzer requires full package installation with all dependencies.\n")
  cat("Alternative: Use the web GUI at https://iohanalyzer.liacs.nl\n")
})

cat("\nFor 100% reproducible analysis with IOHanalyzer:\n")
cat("1. Upload data to https://iohanalyzer.liacs.nl (web GUI)\n")
cat("2. Or install all R dependencies and IOHanalyzer package\n")
cat("3. Data is ready in IOHprofiler format in ioh_data/ directory\n")

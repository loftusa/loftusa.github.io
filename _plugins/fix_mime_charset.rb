# Debug: verify that plugins are loaded when using `jekyll serve` (not hawkins liveserve)
LOG_PATH = "/Users/alex/loftusa.github.io/.cursor/debug.log"

# #region agent log
File.open(LOG_PATH, "a") { |f| f.puts("{\"hypothesisId\":\"H1-verify\",\"location\":\"fix_mime_charset.rb:top\",\"message\":\"Plugin file loaded via jekyll serve\",\"timestamp\":#{(Time.now.to_f * 1000).to_i}}") }
# #endregion

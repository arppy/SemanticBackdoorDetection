# get the avg from ASR output file
grep -oP 'RESULTS: \(\K[^)]+' model_ASR_eval_allbackdoor.out | awk -F',' '{sum += $3; count++} END {print "Mean:", sum/count}'
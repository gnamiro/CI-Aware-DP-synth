BASE_DIR="Compas"



# # dry-run mode (set to false to actually rename)
# DRY_RUN=false

# # Renaming the directiories
# for cs_dir in "$BASE_DIR"/cs=*; do
#   [ -d "$cs_dir" ] || continue
#     tvd_dir="$cs_dir/tvd_L2"
#     privci_dir="$cs_dir/privCI"

#     if [ -d "$tvd_dir" ]; then
#         echo "Found: $tvd_dir"
#         if [ "$DRY_RUN" = false ]; then
#             mv "$tvd_dir" "$privci_dir"
#         fi
#     fi
# done

## Renaming the results files in privCI/eps=10 directories
# DRY_RUN=false   # set to false to actually rename

# # Renaming the results files
# echo "Processing: $BASE_DIR"
# for cs_dir in "$BASE_DIR"/cs=*; do
#     [[ -d "$cs_dir" ]] || continue
#     echo "  CS: $cs_dir"
#     [[ -d "$cs_dir" ]] || continue
#     eps_dir="$cs_dir/privCI/eps=10"
#     [[ -d "$eps_dir" ]] || continue

#     for cv in $(seq 0 4); do
#         src="$eps_dir/results_mst_tvd_L2_same_size_${cv}.csv"
#         dst="$eps_dir/results_privCI_${cv}.csv"
#         echo "    Checking: $src"
#         if [[ -f "$src" ]]; then
#         echo "    Renaming: $src -> $dst"
#         if [[ "$DRY_RUN" = false ]]; then
#             mv -- "$src" "$dst"
#         fi
#         fi
#     done
# done

# # ## Deleting non-results files in privCI/eps=10 directories
# DRY_RUN=false   # set to false to actually rename

# keep_regex='^[Rr]esults_[Pp]riv[Cc][Ii]_[0-9]+\.csv$'
# # keep_regex='^[Rr]esults_[Gg]reedy_[0-9]+\.csv$'
# # keep_regex='^[Rr]esults_[Oo]pt_[0-9]+\.csv$'
# # keep_regex='^[Rr]esults_[Mm]st_[0-9]+\.csv$'

# for cs_dir in "$BASE_DIR"/cs=*; do
#   [[ -d "$cs_dir" ]] || continue
#     [[ -d "$cs_dir" ]] || continue
#     eps_dir="$cs_dir/privCI/eps=0.1"
#     [[ -d "$eps_dir" ]] || continue

#     echo "Processing $eps_dir"
#     shopt -s nullglob
#     for file in "$eps_dir"/*; do
#       [[ -f "$file" ]] || continue
#       fname=$(basename -- "$file")

#       # Only consider 'results_' files; ignore anything else just in case
#       if [[ $fname != results_* ]]; then
#         echo "  (skip non-results file) $fname"
#         continue
#       fi

#       if [[ $fname =~ $keep_regex ]]; then
#         # strict keep for results_privCI_<cv>.csv (case-insensitive)
#         echo "  Keeping:  $file"
#       else
#         echo "  Deleting: $file"
#         if [[ "$DRY_RUN" = false ]]; then
#           rm -f -- "$file"
#         fi
#       fi
#     done
#   done
# done

# # This script will delete directories named otclean, MF, KNN, cmi in each
# DRY_RUN=false 

# for cs_dir in "$BASE_DIR"/cs=*; do
#   [ -d "$cs_dir" ] || continue

#     for sub in hard_constraint otclean ours_cmi frac=0 frac=1 cmi ours_tvd_L1 ours_tvd_L2 vanilla_greedy vanilla_opt vanilla_mst; do
#       target="$cs_dir/$sub"
#       if [ -d "$target" ]; then
#         echo "Deleting dir: $target"
#         if [ "$DRY_RUN" = false ]; then
#           rm -rf -- "$target"
#         fi
#       fi
#     done
#   done
# done

# # # ## Deleting non-results files in privCI/eps=10 directories
DRY_RUN=false   # set to false to actually rename


for cs_dir in "$BASE_DIR"/cs=*; do
  [[ -d "$cs_dir" ]] || continue
    [[ -d "$cs_dir" ]] || continue
    eps_dir="$cs_dir/mst"
    [[ -d "$eps_dir" ]] || continue

    echo "Processing $eps_dir"
    shopt -s nullglob
    for file in "$eps_dir"/*; do
      [[ -f "$file" ]] || continue
      fname=$(basename -- "$file")

      # Only consider 'results_' files; ignore anything else just in case
      if [[ $fname != results_* ]]; then
        echo "  (skip non-results file) $fname"
        continue
      fi
    echo "  Deleting: $file"
    if [[ "$DRY_RUN" = false ]]; then
        rm -f -- "$file"
    fi
    done
  done
done
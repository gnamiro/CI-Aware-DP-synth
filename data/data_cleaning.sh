#!/bin/bash

BASE_DIR="Car_indirect"



# # dry-run mode (set to false to actually rename)
# DRY_RUN=false

# # Renaming the directiories
# for cs_dir in "$BASE_DIR"/cs=*; do
#   [ -d "$cs_dir" ] || continue
#   for frac_dir in "$cs_dir"/frac=*; do
#     [ -d "$frac_dir" ] || continue
#     tvd_dir="$frac_dir/cmi"
#     privci_dir="$frac_dir/privCI"

#     if [ -d "$tvd_dir" ]; then
#         echo "Found: $tvd_dir"
#         if [ "$DRY_RUN" = false ]; then
#             mv "$tvd_dir" "$privci_dir"
#         fi
#     fi
#   done
# done


# ## Renaming the results files in privCI/eps=10 directories
# DRY_RUN=false   # set to false to actually rename

# # Renaming the results files
# echo "Processing: $BASE_DIR"
# for cs_dir in "$BASE_DIR"/cs=*; do
#   [[ -d "$cs_dir" ]] || continue
#   echo "  CS: $cs_dir"
#   for frac_dir in "$cs_dir"/frac=*; do
#     [[ -d "$frac_dir" ]] || continue
#     eps_dir="$frac_dir/privCI/eps=10"
#     [[ -d "$eps_dir" ]] || continue

#     for cv in $(seq 0 4); do
#       src="$eps_dir/results_mst_cmi_weighted_optim_${cv}.csv"
#       dst="$eps_dir/results_privCI_${cv}.csv"
#       echo "    Checking: $src"
#       if [[ -f "$src" ]]; then
#         echo "    Renaming: $src -> $dst"
#         if [[ "$DRY_RUN" = false ]]; then
#           mv -- "$src" "$dst"
#         fi
#       fi
#     done
#   done
# done



# ## Deleting non-results files in privCI/eps=10 directories
DRY_RUN=false   # set to false to actually rename

keep_regex='^[Rr]esults_[Pp]riv[Cc][Ii]_[0-9]+\.csv$'
# keep_regex='^[Rr]esults_[Gg]reedy_[0-9]+\.csv$'

for cs_dir in "$BASE_DIR"/cs=*; do
  [[ -d "$cs_dir" ]] || continue
  for frac_dir in "$cs_dir"/frac=*; do
    [[ -d "$frac_dir" ]] || continue
    eps_dir="$frac_dir/privCI/eps=10"
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

      if [[ $fname =~ $keep_regex ]]; then
        # strict keep for results_privCI_<cv>.csv (case-insensitive)
        echo "  Keeping:  $file"
      else
        echo "  Deleting: $file"
        if [[ "$DRY_RUN" = false ]]; then
          rm -f -- "$file"
        fi
      fi
    done
  done
done


# # This script will delete directories named otclean, MF, KNN, cmi in each
# DRY_RUN=false 

# for cs_dir in "$BASE_DIR"/cs=*; do
#   [ -d "$cs_dir" ] || continue
#   for frac_dir in "$cs_dir"/frac=*; do
#     [ -d "$frac_dir" ] || continue

#     for sub in otclean MF KNN cmi; do
#       target="$frac_dir/$sub"
#       if [ -d "$target" ]; then
#         echo "Deleting dir: $target"
#         if [ "$DRY_RUN" = false ]; then
#           rm -rf -- "$target"
#         fi
#       fi
#     done
#   done
# done

# DRY_RUN=false 
# for cs_dir in "$BASE_DIR"/cs=*; do
#   [ -d "$cs_dir" ] || continue
#   for frac_dir in "$cs_dir"/frac=*; do
#     [ -d "$frac_dir" ] || continue
#     privCI="$frac_dir/greedy"
#     [[ -d "$privCI" ]] || continue

#     for sub in 0.1 0.5; do
#       target="$privCI/eps=$sub"
#       if [ -d "$target" ]; then
#         echo "Deleting dir: $target"
#         if [ "$DRY_RUN" = false ]; then
#           rm -rf -- "$target"
#         fi
#       fi
#     done
#   done
# done


if [ "$DRY_RUN" = true ]; then
  echo "Dry-run: no changes made. Set DRY_RUN=false in script to apply."
fi

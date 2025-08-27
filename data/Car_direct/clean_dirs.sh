#!/bin/bash

# Navigate through each cross-validation directory (cs=0, cs=1, ...)
for cs_dir in cs=*; do
  # Check if it's a directory
  [ -d "$cs_dir" ] || continue

  # Within each cs directory, go through each frac directory
  for frac_dir in "$cs_dir"/frac=*; do
    [ -d "$frac_dir" ] || continue

    # Try removing KNN and MF if they exist
    for subdir in KNN MF; do
      target="$frac_dir/$subdir"
      if [ -d "$target" ]; then
        rm -r "$target"
        echo "Removed $target"
      fi
    done
  done
done


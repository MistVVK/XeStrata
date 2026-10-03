#!/bin/sh
# READY, then: ERR for an image named *bad*, exit for *die*, no answer for *hang*, else OK
echo "READY 2560"
while read cmd img out; do
  case "$img" in
    *bad*) echo "ERR cannot read the image $img" ;;
    *die*) exit 3 ;;
    *hang*) sleep 30 ;;
    *) printf 'SVE1' > "$out"; echo "OK 4 2 2 1" ;;
  esac
done

#!/usr/bin/env bash
# 8 连接分段下载（curl 原生进程，CPU 占用极低；用于 hf-mirror 单流被限速时）
#   bash scripts/_curl_parts.sh <url> <out> <size_bytes> [parts]
set -u
URL="$1"; OUT="$2"; SIZE="$3"; PARTS="${4:-8}"
DIR="$(dirname "$OUT")"; BASE="$(basename "$OUT")"
mkdir -p "$DIR"
STEP=$(( SIZE / PARTS ))
PIDS=()
for i in $(seq 0 $(( PARTS - 1 ))); do
  START=$(( i * STEP ))
  if [ "$i" -eq $(( PARTS - 1 )) ]; then END=$(( SIZE - 1 )); else END=$(( START + STEP - 1 )); fi
  PART="$DIR/.$BASE.p$i"
  if [ -f "$PART" ] && [ "$(stat -c%s "$PART")" -eq $(( END - START + 1 )) ]; then
    echo "[skip] part$i"; continue
  fi
  curl -sS -L -A "Mozilla/5.0" -r "$START-$END" -o "$PART" "$URL" --retry 5 --retry-delay 3 &
  PIDS+=($!)
done
for p in "${PIDS[@]:-}"; do wait "$p"; done
TOTAL=0
for i in $(seq 0 $(( PARTS - 1 ))); do
  P="$DIR/.$BASE.p$i"; S=$(stat -c%s "$P" 2>/dev/null || echo 0); TOTAL=$(( TOTAL + S ))
done
echo "[dl] parts total $TOTAL / $SIZE"
if [ "$TOTAL" -ne "$SIZE" ]; then echo "[dl] 未完整，保留分段可重跑"; exit 2; fi
: > "$OUT"
for i in $(seq 0 $(( PARTS - 1 ))); do cat "$DIR/.$BASE.p$i" >> "$OUT"; rm -f "$DIR/.$BASE.p$i"; done
echo "[dl] ok $OUT $(stat -c%s "$OUT") bytes"

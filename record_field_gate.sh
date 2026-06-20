#!/usr/bin/env bash
set -euo pipefail

POS_DIR="data/validation/positive"
NEG_DIR="data/validation/negative"
COUNT="${1:-20}"

mkdir -p "$POS_DIR" "$NEG_DIR"

last_index() {
  find "$1" -maxdepth 1 -name "$2-*.wav" -print \
    | sed -E "s/.*$2-([0-9]+)\.wav/\1/" \
    | sort -n \
    | tail -1
}

LAST_POS="$(last_index "$POS_DIR" auvin)"
LAST_NEG="$(last_index "$NEG_DIR" bg)"
LAST_POS="${LAST_POS:-0}"
LAST_NEG="${LAST_NEG:-0}"
START=$((10#$LAST_POS > 10#$LAST_NEG ? 10#$LAST_POS + 1 : 10#$LAST_NEG + 1))
END=$((START + COUNT - 1))

echo "============================================"
echo "  Auvin Wake Word — Field Gate Recording"
echo "  Pronunciation: Aww-win (like 'awning' -ing)"
echo "  Recording $COUNT positive + $COUNT negative clips"
echo "  Appending clip indices $START through $END"
echo "============================================"
echo ""

# --- Positives ---
echo ">>>> POSITIVES: Say \"Auvin\" or \"Hey Auvin\" clearly."
echo "    Each 4-second clip starts after a short ready cue."
echo ""
record_number=0
for index in $(seq "$START" "$END"); do
  record_number=$((record_number + 1))
  i=$(printf "%02d" "$index")
  f="$POS_DIR/auvin-$i.wav"
  echo "[$record_number/$COUNT] Get ready: say AWW-WIN or HEY AWW-WIN..."
  sleep 1
  echo "  RECORDING NOW"
  rec -r 16000 -c 1 -b 16 "$f" trim 0 4 2>/dev/null
  echo "  Saved $f"
done

echo ""
echo ">>>> POSITIVES DONE ($COUNT clips)"
echo ""

# --- Negatives ---
echo ">>>> NEGATIVES: Record background noise, conversation,"
echo "    TV, music, silence — anything except the wake word."
echo "    Each clip is 10 seconds."
echo ""
record_number=0
for index in $(seq "$START" "$END"); do
  record_number=$((record_number + 1))
  i=$(printf "%02d" "$index")
  f="$NEG_DIR/bg-$i.wav"
  echo "[$record_number/$COUNT] Recording 10s of background..."
  rec -r 16000 -c 1 -b 16 "$f" trim 0 10 2>/dev/null
  echo "  Saved $f"
done

echo ""
echo ">>>> NEGATIVES DONE ($COUNT clips)"
echo ""

# --- Validate with the production Node detector (VAD + debounce + cooldown) ---
echo "============================================"
echo "  Running validation..."
echo "============================================"
npm run validate:field --workspace=@auvin/wake-word

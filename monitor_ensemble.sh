#!/bin/bash
# Monitor ensemble optimization and send summary when complete

LOG_FILE="ensemble_optimization.log"
RESULTS_FILE="optimization_wf_ensemble_results.csv"
CHECK_INTERVAL=300  # 5 minutes

while true; do
    # Check if ensemble is still running
    if ! ps aux | grep -q "optimize_wf.*ensemble.*200" | grep -v grep; then
        # Process finished, check results
        if [ -f "$RESULTS_FILE" ]; then
            echo "✅ ENSEMBLE OPTIMIZATION COMPLETE"
            echo ""
            echo "Top 10 configurations:"
            head -11 "$RESULTS_FILE" | column -t -s,

            # Calculate summary
            TOTAL_LINES=$(wc -l < "$RESULTS_FILE")
            POSITIVE_SHARPE=$(awk -F',' '$8 > 0 {count++} END {print count}' "$RESULTS_FILE")
            SHARPE_GT_1=$(awk -F',' '$8 > 1.0 {count++} END {print count}' "$RESULTS_FILE")

            echo ""
            echo "Summary:"
            echo "  Total trials: $((TOTAL_LINES - 1))"
            echo "  Positive Sharpe: $POSITIVE_SHARPE"
            echo "  Sharpe > 1.0: $SHARPE_GT_1"

            # Exit
            exit 0
        fi
    fi

    # Get current progress
    CURRENT=$(tail -1 "$LOG_FILE" | grep -oP '\d+(?=/200)')
    if [ -n "$CURRENT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Progress: $CURRENT/200"
    fi

    sleep $CHECK_INTERVAL
done

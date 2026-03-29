#!/bin/bash
# Watchdog: restart services that stop producing output
# Checks log freshness (journald or log file) AND memory as a combined signal.
#
# For most services: checks journald.
# For diamond-monitor: checks the log FILE (it uses StandardOutput=append, not journald).
#
# A service is restarted if its log output is stale beyond max_age_min.
# Zombie detection (stale + low memory) triggers a more urgent message.

NOW=$(date +%s)

check_service() {
    local svc=$1 max_age_min=$2 logfile=${3:-}

    # Check if systemd thinks it's running
    if ! systemctl is-active --quiet "$svc"; then
        echo "[$svc] DOWN — systemd will auto-restart"
        return
    fi

    # Get memory usage (bytes)
    local mem_bytes
    mem_bytes=$(systemctl show "$svc" --property=MemoryCurrent 2>/dev/null | cut -d= -f2)

    # Get last log timestamp — from file or journald
    local last_ts=0
    if [ -n "$logfile" ] && [ -f "$logfile" ]; then
        # Use file modification time (updated on every write)
        last_ts=$(stat -c %Y "$logfile" 2>/dev/null || echo 0)
    else
        last_ts=$(journalctl -u "$svc" -n 1 --output=short-unix 2>/dev/null | tail -1 | awk '{print int($1)}')
    fi

    # Compute age in minutes (0 if timestamp unavailable)
    local age_min=0
    if [ -n "$last_ts" ] && [ "$last_ts" -gt 0 ] 2>/dev/null; then
        age_min=$(( (NOW - last_ts) / 60 ))
    fi

    local is_stale=false
    if [ "$age_min" -gt "$max_age_min" ]; then
        is_stale=true
    fi

    # A true zombie: stale output AND very low memory (< 2 MB)
    # Normal sleeping Python: 3-50 MB with recent log output
    local is_zombie_mem=false
    if [ -n "$mem_bytes" ] && [ "$mem_bytes" != "[not set]" ] && [ "$mem_bytes" -lt 2000000 ] 2>/dev/null; then
        is_zombie_mem=true
    fi

    if $is_stale && $is_zombie_mem; then
        echo "[$svc] ZOMBIE: stale ${age_min}min + memory ${mem_bytes}B — restarting"
        logger -t watchdog "Restarting $svc: zombie (stale=${age_min}min, mem=${mem_bytes}B)"
        sudo systemctl restart "$svc"
    elif $is_stale; then
        echo "[$svc] STALE: last output ${age_min}min ago (max ${max_age_min}) — restarting"
        logger -t watchdog "Restarting $svc: stale journal (${age_min}min > ${max_age_min}min)"
        sudo systemctl restart "$svc"
    else
        local mem_mb="?"
        if [ -n "$mem_bytes" ] && [ "$mem_bytes" != "[not set]" ] 2>/dev/null; then
            mem_mb=$(( mem_bytes / 1048576 ))
        fi
        echo "[$svc] OK (${age_min}min since last output, ${mem_mb}MB, max ${max_age_min}min)"
    fi
}

# AGATE: scans every 240min, allow 360min (scan + 2h buffer)
check_service agate-trader    360
# BERYL: daily scans (equities), allow 26h = 1560min
check_service beryl-trader    1560
# CITRINE: daily scans, allow 26h = 1560min
check_service citrine-trader  1560
# DIAMOND: WebSocket streaming, should log every 2min; allow 10min staleness
# Uses log FILE (not journald) — Diamond uses StandardOutput=append
check_service diamond-monitor 10 /home/ubuntu/kalshi-diamond/diamond_monitor.log

# DASHBOARDS: Dash apps log on startup + every auto-refresh (60s).
# Allow 10min staleness — if no log output for 10min, the dashboard is dead.
check_service live-dashboard          10
check_service citrine-dashboard       10
check_service diamond-dashboard       10
check_service consolidated-dashboard  10

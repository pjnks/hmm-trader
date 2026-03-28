#!/bin/bash
# 5-min accountability check-in — interactive dialogs that REQUIRE acknowledgment
# Triggered by cron at 9:30pm (when user is at MacBook)

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
/Users/perryjenkins/opt/anaconda3/bin/python daily_report.py --checkin

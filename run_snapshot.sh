#!/bin/bash
# Midnight maturity snapshot — saves daily scores to maturity_history.csv
# Triggered by cron at 12:00am

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
/Users/perryjenkins/opt/anaconda3/bin/python daily_report.py --snapshot >> /Users/perryjenkins/Documents/trdng/HMM-Trader/daily_report.log 2>&1

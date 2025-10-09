# Traffic Data Branch

This branch contains automated traffic data collection for the UVisBox repository.

## Data Structure
- `traffic_data/traffic_YYYY-MM-DD.json`: Daily traffic data files
- `traffic_data/latest_summary.json`: Latest traffic summary

## Data Contents
Each daily file contains:
- **Views**: Page views and unique visitors
- **Clones**: Repository clones and unique cloners  
- **Referrers**: Top referral sources
- **Popular Paths**: Most visited repository paths

## Automation
Data is automatically updated daily at midnight UTC via GitHub Actions workflow.

## Branch Purpose
This branch keeps traffic data separate from the main codebase to maintain a clean development environment while preserving valuable analytics data.

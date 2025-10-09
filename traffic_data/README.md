# Repository Traffic Data

This directory contains automated traffic data collection from the GitHub repository.

## Overview

The traffic data is collected daily at midnight UTC using a GitHub Action workflow. The data includes:

- **Repository Views**: Total views and unique visitors over the past 14 days
- **Repository Clones**: Total clones and unique cloners over the past 14 days
- **Top Referrers**: Sources that direct traffic to the repository
- **Popular Paths**: Most visited files and directories

## Files

- `traffic_YYYY-MM-DD.json`: Daily traffic data snapshots
- `latest_summary.json`: Most recent traffic data summary

## Data Structure

Each traffic data file contains:

```json
{
  "timestamp": "2024-XX-XXTXX:XX:XX.XXXXXXX+00:00",
  "date": "YYYY-MM-DD",
  "repository": "owner/repository-name",
  "data": {
    "views": {
      "count": 123,
      "uniques": 45,
      "views": [
        {
          "timestamp": "2024-XX-XXTXX:XX:XX.XXXXXXX+00:00",
          "count": 10,
          "uniques": 5
        }
      ]
    },
    "clones": {
      "count": 67,
      "uniques": 23,
      "clones": [
        {
          "timestamp": "2024-XX-XXTXX:XX:XX.XXXXXXX+00:00",
          "count": 5,
          "uniques": 3
        }
      ]
    },
    "referrers": [
      {
        "referrer": "github.com",
        "count": 15,
        "uniques": 8
      }
    ],
    "popular_paths": [
      {
        "path": "/owner/repository/blob/main/README.md",
        "title": "README.md",
        "count": 25,
        "uniques": 12
      }
    ]
  }
}
```

## Automation

The data collection is automated through the GitHub Action workflow `.github/workflows/traffic-data-collection.yml`. 

### Schedule
- Runs daily at midnight UTC (00:00)
- Can be manually triggered via GitHub Actions tab

### Permissions Required
- `contents: write` - To commit and push traffic data files
- `repository-projects: read` - To access repository traffic analytics

## Manual Execution

To manually run the traffic data collection:

1. Go to the "Actions" tab in your GitHub repository
2. Select "Repository Traffic Data Collection" workflow
3. Click "Run workflow" button
4. Choose the branch and click "Run workflow"

## Data Retention

- GitHub provides traffic data for the past 14 days
- This automation preserves historical data beyond the 14-day limit
- Daily snapshots allow for long-term trend analysis

## Privacy and Usage

This data collection only accesses publicly available repository traffic analytics that are already visible to repository owners in the GitHub interface. No personal user information is collected or stored.

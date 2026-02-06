# 📊 Steam Data Collection Pipeline

This module provides a comprehensive ETL (Extract, Transform, Load) pipeline for collecting Steam game data and reviews directly from Steam's public APIs. It's designed for robust, large-scale data collection with resume capabilities and error handling.

## 🎯 Overview

The data collection pipeline consists of several scripts that work together to:
1. **Fetch all Steam app IDs** from Steam's public API
2. **Collect detailed app information and reviews** for each game
3. **Clean up incomplete data directories**
4. **Monitor and control long-running processes**

## 📁 Pipeline Components

### 1. `getAppIds.py` - Steam App ID Collection
**Purpose**: Fetches and sorts all available Steam application IDs

**Features**:
- Retrieves complete Steam app catalog from Steam Web API
- Sorts apps by app ID for consistent processing order
- Saves results in structured JSON format
- Handles API errors gracefully

**Usage**:
```bash
python getAppIds.py
```

**Output**: `../appList/sorted_steam_apps.json`

---

### 2. `getAppDetails.py` - Main Data Collection Engine
**Purpose**: Comprehensive Steam app data and review collection with resume capabilities

**Key Features**:
- **Resume Capability**: Can restart from where it left off if interrupted
- **Rate Limiting**: Built-in delays to respect Steam's API limits
- **Progress Tracking**: Maintains resume points and detailed logging
- **Robust Error Handling**: Continues processing even if individual apps fail
- **Incremental Saving**: Saves progress after each review page

**Data Collected**:
- Game metadata (name, app ID, etc.)
- Complete review history with pagination
- Author information and review statistics
- Review language and sentiment indicators

**Usage**:
```bash
python getAppDetails.py
```

**Resume Mechanism**:
The script automatically resumes from interruptions using:
- `resume_point.txt`: Tracks which app ID to start from
- `rt_data` files: Store pagination cursors for individual apps
- Progress logging for monitoring

---

### 3. `indie_devs_Details.py` - Targeted Collection
**Purpose**: Focused data collection for specific indie games

**Features**:
- Processes predefined list of app IDs
- Same robust collection logic as main script
- Ideal for targeted analysis or testing
- Separate output directory to avoid conflicts

**Predefined Games**:
- 1794680, 413150, 367520, 105600 (customizable)

**Usage**:
```bash
python indie_devs_Details.py
```

---

### 4. `clean_directories.py` - Data Cleanup Utility
**Purpose**: Removes empty directories from failed collection attempts

**Safety Features**:
- **Dry Run Mode**: Preview what will be deleted before actual deletion
- **Confirmation Prompt**: Requires explicit "YES" to proceed
- **Detailed Logging**: Logs all actions and decisions
- **Error Handling**: Continues processing even if individual operations fail

**Usage**:
```bash
python clean_directories.py
```

**Safety Protocol**:
1. Shows target directory path
2. Requires "YES" confirmation
3. Logs all operations to `log.txt`
4. Creates `non_empty_dirs.txt` with preserved directories

---

### 5. `kill_pid.py` - Process Monitoring Tool
**Purpose**: Automatically terminates data collection at specific points

**Use Cases**:
- Stop collection after reaching specific app IDs
- Automated testing and debugging
- Resource management for long-running collections

**Features**:
- Real-time log monitoring
- Process identification by name or command line
- Graceful termination when trigger phrase appears

**Usage**:
```bash
python kill_pid.py
```

**Configuration**:
```python
log_file_path = "../app_details_json/steam_app_scraper.log"
target_phrase = "Processing appID:"  # Customize trigger
```

## 🚀 Quick Start Guide

### Step 1: Collect All Steam App IDs
```bash
cd src/Step_1_collection_and_cleaning/
python getAppIds.py
```
**Expected Output**: `../appList/sorted_steam_apps.json` with ~100,000+ app IDs

### Step 2: Start Full Data Collection
```bash
python getAppDetails.py
```
**Note**: This process can take days/weeks for complete Steam catalog

### Step 3: Monitor Progress
- Check `../app_details_json/steam_app_scraper.log` for real-time progress
- Resume point saved in `../app_details_json/resume_point.txt`
- Individual app progress in `{appID}_data/rt_data` files

### Step 4: Clean Up (if needed)
```bash
python clean_directories.py
```

## 📋 Data Structure

### Output Directory Structure
```
../app_details_json/
├── steam_app_scraper.log           # Main log file
├── resume_point.txt                # Global resume position
└── {appID}_data/                   # Per-app directories
    ├── Review_Details{appID}.json  # Complete app data + reviews
    └── rt_data                     # Resume data (cursor positions)
```

### JSON Data Format
```json
{
  "appID": {
    "data": {
      "name": "Game Name",
      "steam_appid": 123456,
      "review_stats": {
        "total_num_reviews": 1500,
        "reviews": [
          {
            "review": "Review text...",
            "voted_up": true,
            "language": "english",
            "author": {...},
            "votes_up": 5,
            "weighted_vote_score": 0.8
          }
        ]
      }
    }
  }
}
```

## ⚙️ Configuration Options

### Rate Limiting
```python
time.sleep(0.1)  # 100ms between API calls
time.sleep(0.5)  # 500ms between apps
```

### Collection Parameters
```python
num_of_reviews_per_page = 100  # Reviews per API call
skip_if_exists = True          # Skip already processed apps
```

### Resume Behavior
- **Automatic**: Script detects incomplete collections and resumes
- **Manual**: Edit `resume_point.txt` to start from specific app index
- **Per-App**: Individual apps resume from last successful page

## 🔧 Error Handling & Recovery

### Common Issues & Solutions

**API Rate Limiting**:
- Built-in delays prevent most rate limiting
- If blocked, wait and restart - resume will continue from last position

**Network Interruptions**:
- Resume mechanism handles disconnections gracefully
- Check logs for last successful operation

**Disk Space**:
- Monitor available space (large datasets can be 100GB+)
- Consider processing in batches using app ID ranges

**Memory Issues**:
- Script processes apps individually to minimize memory usage
- Clean up empty directories periodically

### Recovery Commands
```bash
# Check resume status
cat ../app_details_json/resume_point.txt

# View recent log entries
tail -50 ../app_details_json/steam_app_scraper.log

# Clean up failed attempts
python clean_directories.py
```

## 📊 Performance Considerations

### Expected Runtime
- **App IDs Collection**: ~30 seconds
- **Full Steam Catalog**: 2-4 weeks (depending on network/API limits)
- **Targeted Collection**: Minutes to hours per app

### Resource Usage
- **Network**: Moderate (respects API rate limits)
- **Disk**: High (can reach 100GB+ for full catalog)
- **Memory**: Low (processes one app at a time)
- **CPU**: Low (mostly I/O bound)

### Optimization Tips
1. **Run on stable network connection**
2. **Use SSD for faster I/O operations**
3. **Monitor disk space regularly**
4. **Consider collecting specific categories first**

## 🛡️ Best Practices

### Before Starting
1. **Test with indie_devs_Details.py** first
2. **Ensure adequate disk space** (100GB+ recommended)
3. **Verify network stability**
4. **Set up monitoring** for long runs

### During Collection
1. **Monitor logs regularly** for errors
2. **Check disk space periodically**
3. **Don't interrupt unnecessarily** (resume adds overhead)
4. **Keep system stable** (avoid reboots/updates)

### After Collection
1. **Run clean_directories.py** to remove failed attempts
2. **Backup collected data** before processing
3. **Verify data integrity** with sample checks
4. **Document collection parameters** for reproducibility

## 🔍 Troubleshooting

### Script Won't Start
```bash
# Check Python environment
python --version
pip list | grep requests

# Verify file paths
ls -la ../appList/
ls -la ../app_details_json/
```

### Collection Stalls
```bash
# Check current app being processed
tail -10 ../app_details_json/steam_app_scraper.log

# Verify network connectivity
curl -I "https://store.steampowered.com/api/AppList/"

# Check disk space
df -h
```

### Resume Issues
```bash
# Check resume point
cat ../app_details_json/resume_point.txt

# Reset to specific position (if needed)
echo "1000" > ../app_details_json/resume_point.txt
```

## 📝 Legal & Ethical Considerations

- **Respects Steam's ToS**: Uses only public APIs
- **Rate Limited**: Built-in delays prevent server overload
- **Public Data Only**: Collects only publicly available reviews
- **Research Purpose**: Intended for academic/research use
- **Attribution**: Consider citing Steam as data source

## 🤝 Contributing

When modifying the collection scripts:
1. **Maintain resume capability** for long-running processes
2. **Add comprehensive logging** for debugging
3. **Test with small datasets** before full runs
4. **Document configuration changes**
5. **Preserve backward compatibility** with existing data

---

**⚠️ Important**: This pipeline can generate large amounts of data. Ensure you have adequate storage and network resources before starting a full collection.
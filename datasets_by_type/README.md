# Datasets Organized by Type

This directory contains all UX datasets organized by **dataset type**, with sample sizes as subdirectories within each type.

## Directory Structure

```
datasets_by_type/
├── 01_survey_questionnaire/
│   ├── small/
│   │   └── survey_questionnaire_small.csv
│   ├── medium/
│   │   └── survey_questionnaire_medium.csv
│   └── large/
│       └── survey_questionnaire_large.csv
├── 02_usability_testing/
│   ├── small/
│   ├── medium/
│   └── large/
├── ... (and so on for all 17 dataset types)
```

## Dataset Types

### 01. Survey & Questionnaire Data
- **Description**: Post-task surveys with SUS, UEQ, NASA-TLX, Trust measures
- **Sample sizes**: 50 (small), 200 (medium), 1000 (large)
- **Variables**: 27 including demographics, task performance, validated scales

### 02. Usability Testing
- **Description**: Lab-based usability testing with task completion metrics
- **Sample sizes**: 50, 200, 1000 task observations
- **Variables**: 11 including success, errors, satisfaction, device type

### 03. Interaction / Telemetry Logs
- **Description**: Web interaction clickstream and event tracking
- **Sample sizes**: ~500, 2000, 10000 events
- **Variables**: 16 including clicks, scrolls, conversions, device info

### 04. Eye Tracking
- **Description**: Eye tracking across areas of interest
- **Sample sizes**: 50, 200, 1000 AOI observations
- **Variables**: 15 including fixations, dwell time, TTFF, pupil diameter

### 05. Physiological Data (EEG/GSR)
- **Description**: Psychophysiological measures during engagement
- **Sample sizes**: 50, 200, 1000 measurements
- **Variables**: 17 including EEG, GSR, heart rate, engagement

### 06. Card Sorting
- **Description**: Information architecture categorization study
- **Sample sizes**: 50, 200, 1000 card categorizations
- **Variables**: 10 including correctness, confidence, path length

### 07. A/B Testing
- **Description**: A/B test with conversion and engagement metrics
- **Sample sizes**: 50, 200, 1000 users
- **Variables**: 10 including conversion, revenue, bounce rate

### 08. Funnel / Retention
- **Description**: Multi-step conversion funnel with drop-off tracking
- **Sample sizes**: 50, 200, 1000 users
- **Variables**: 10 including step completions, retention, exit points

### 09. Search / Information Architecture
- **Description**: Search query logs with reformulation and clicks
- **Sample sizes**: ~500, 2000, 10000 search queries
- **Variables**: 12 including queries, clicks, reformulations

### 10. Chatbot / Conversational
- **Description**: Voice assistant/conversational AI interaction data
- **Sample sizes**: ~400, 1600, 8000 conversation turns
- **Variables**: 13 including intents, sentiment, latency, escalation

### 11. Accessibility Testing
- **Description**: Usability testing with assistive technologies
- **Sample sizes**: 50, 200, 1000 participants
- **Variables**: 12 including disability type, assistive tech, WCAG issues

### 12. Cognitive Task Data
- **Description**: Experimental cognitive task (Stroop-like) with RT/accuracy
- **Sample sizes**: ~2400, 9600, 48000 trials
- **Variables**: 14 including conditions, RT, accuracy, demographics

### 13. Game Analytics
- **Description**: Player telemetry and engagement in games
- **Sample sizes**: 50, 200, 1000 gaming sessions
- **Variables**: 13 including score, XP, deaths, engagement

### 14. Spatial / XR Data
- **Description**: AR/VR spatial interaction and gaze tracking
- **Sample sizes**: 50, 200, 1000 interactions
- **Variables**: 13 including 3D coordinates, gaze, latency

### 15. Diary / Qualitative
- **Description**: Remote diary study with sentiment analysis
- **Sample sizes**: 50, 200, 1000 entries
- **Variables**: 11 including text, sentiment, themes, emotions

### 16. Feature Adoption
- **Description**: Feature usage and retention time series
- **Sample sizes**: 50, 200, 1000 users
- **Variables**: 14 including usage, retention days, engagement

### 17. System UX Metrics
- **Description**: Integrated telemetry combining UX and performance
- **Sample sizes**: 50, 200, 1000 task completions
- **Variables**: 15 including success, latency, satisfaction, errors

## Usage

### Access by Type and Size
```python
import pandas as pd

# Load specific dataset by type and size
df = pd.read_csv('datasets_by_type/01_survey_questionnaire/large/survey_questionnaire_large.csv')
```

### Find All Datasets of a Type
All sizes for survey questionnaires are in: `01_survey_questionnaire/small/`, `medium/`, `large/`

### Compare Different Sizes of Same Type
```python
# Load all sizes of eye tracking data
small = pd.read_csv('datasets_by_type/04_eye_tracking/small/eye_tracking_small.csv')
medium = pd.read_csv('datasets_by_type/04_eye_tracking/medium/eye_tracking_medium.csv')
large = pd.read_csv('datasets_by_type/04_eye_tracking/large/eye_tracking_large.csv')
```

## Benefits of This Organization

1. **Type-Based**: Easy to find all datasets related to a specific method (e.g., surveys, eye tracking)
2. **Size-Hierarchical**: Within each type, all sample sizes are together
3. **Scalable**: Easy to add new datasets or new sizes
4. **Consistent**: Same structure across all dataset types
5. **Teaching-Friendly**: Instructors can focus on specific methods and use appropriate sample sizes

## Documentation

For detailed documentation on each dataset type, see the `docs/` directory in the parent folder:
- `docs/survey_questionnaire_README.md`
- `docs/usability_testing_README.md`
- etc.

## Quick Reference

| Dataset Type | Directory | Small | Medium | Large |
|--------------|-----------|-------|--------|-------|
| Survey | `01_survey_questionnaire` | 50 | 200 | 1000 |
| Usability | `02_usability_testing` | 50 | 200 | 1000 |
| Telemetry | `03_interaction_telemetry` | ~500 | ~2000 | ~10000 |
| Eye Tracking | `04_eye_tracking` | 50 | 200 | 1000 |
| Physiological | `05_physiological` | 50 | 200 | 1000 |
| Card Sorting | `06_card_sorting` | 50 | 200 | 1000 |
| A/B Testing | `07_ab_testing` | 50 | 200 | 1000 |
| Funnel | `08_funnel_retention` | 50 | 200 | 1000 |
| Search | `09_search_ia` | ~500 | ~2000 | ~10000 |
| Chatbot | `10_chatbot_conversational` | ~400 | ~1600 | ~8000 |
| Accessibility | `11_accessibility` | 50 | 200 | 1000 |
| Cognitive | `12_cognitive_task` | ~2400 | ~9600 | ~48000 |
| Game Analytics | `13_game_analytics` | 50 | 200 | 1000 |
| Spatial/XR | `14_spatial_xr` | 50 | 200 | 1000 |
| Diary | `15_diary_qualitative` | 50 | 200 | 1000 |
| Feature Adoption | `16_feature_adoption` | 50 | 200 | 1000 |
| System UX | `17_system_ux_metrics` | 50 | 200 | 1000 |

---

**Total Files**: 51 CSV files (17 types × 3 sizes)  
**Organization**: Type-first, Size-second


# UX Datasets Collection

[![License](https://img.shields.io/badge/License-Educational-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-brightgreen.svg)](https://www.python.org/)
[![Datasets](https://img.shields.io/badge/Datasets-17-orange.svg)]()
[![Sample Sizes](https://img.shields.io/badge/Variants-51%20CSVs-lightgrey.svg)]()

> A comprehensive collection of 17 realistic, multi-method UX research datasets for teaching UX research, HCI, and cognitive psychology.

**Created by [Mohsen Rafiei, Ph.D.](https://github.com/mohsen-rafiei)**  
**Perceptual User Experience Lab (PUX Lab)** - [www.puxlab.com](https://www.puxlab.com/)

---

## Overview

This repository contains synthetically generated but highly realistic UX datasets covering a wide range of research methods and study types. Each dataset is available in **3 sample size variants** (small, medium, large), totaling **51 CSV files** across all 17 dataset types.

The datasets are designed to mirror real-world UX research with:
- Realistic variable distributions (normal, log-normal, Poisson, beta, exponential, etc.)
- Built-in relationships among variables
- Authentic data quality challenges (missing values, outliers, noise)
- Domain-specific metrics aligned with validated instruments (SUS, UEQ, NASA-TLX)
- Multiple research methodologies (surveys, eye tracking, telemetry, etc.)

---

## Use Cases

### For Educators
- Pre-made datasets for assignments and exercises
- Multiple difficulty levels (small → medium → large)
- Demonstrating statistical concepts with realistic data
- Teaching research method comparison

### For Students
- Learning statistical analysis with domain-appropriate data
- Practice with different data types (continuous, categorical, binary, ordinal)
- Understanding data cleaning and preprocessing challenges
- Applying UX metrics and frameworks

### For Researchers
- Validating analysis workflows and tools
- Testing statistical methods and algorithms
- Prototyping analysis pipelines
- Benchmarking and method comparison

---

## Dataset Collection

### Survey & Questionnaire Methods

#### 1. Survey & Questionnaire Data
**Post-task usability surveys with validated instruments**
- `datasets_by_type/01_survey_questionnaire/`
- **Variables**: 27 (SUS, UEQ, NASA-TLX, Trust measures)
- **Sample sizes**: 50 (small), 200 (medium), 1,000 (large)
- **Highlights**: Includes demographics, task performance, and multiple validated scales
- [ Detailed Documentation](docs/survey_questionnaire_README.md)

#### 11. Accessibility Testing
**Usability testing with assistive technologies**
- `datasets_by_type/11_accessibility/`
- **Variables**: 12 (disability type, assistive tech, WCAG issues)
- **Sample sizes**: 50, 200, 1,000 participants
- **Highlights**: Inclusive design research with accessibility barriers
- [ Detailed Documentation](docs/accessibility_README.md)

#### 15. Diary / Qualitative Data
**Remote diary study with sentiment analysis**
- `datasets_by_type/15_diary_qualitative/`
- **Variables**: 11 (text, sentiment, themes, emotions)
- **Sample sizes**: 50, 200, 1,000 entries
- **Highlights**: Mixed-method combining qualitative text with quantitative scores
- [ Detailed Documentation](docs/diary_qualitative_README.md)

### Behavioral & Telemetry Methods

#### 2. Usability Testing
**Lab-based usability testing with direct observation**
- `datasets_by_type/02_usability_testing/`
- **Variables**: 11 (task success, errors, frustration, satisfaction)
- **Sample sizes**: 50, 200, 1,000 task observations
- **Highlights**: Controlled testing environment with multiple tasks
- [ Detailed Documentation](docs/usability_testing_README.md)

#### 3. Interaction / Telemetry Logs
**Web interaction clickstream and event tracking**
- `datasets_by_type/03_interaction_telemetry/`
- **Variables**: 16 (clicks, scrolls, conversions, device info)
- **Sample sizes**: ~500, ~2,000, ~10,000 events
- **Highlights**: Large-scale behavioral data, suitable for segmentation
- [ Detailed Documentation](docs/interaction_telemetry_README.md)

#### 9. Search / Information Architecture
**Search query logs with reformulation behavior**
- `datasets_by_type/09_search_ia/`
- **Variables**: 12 (queries, clicks, reformulations, success)
- **Sample sizes**: ~500, ~2,000, ~10,000 queries
- **Highlights**: Search quality analysis, query patterns
- [ Detailed Documentation](docs/search_ia_README.md)

#### 10. Chatbot / Conversational UX
**Voice assistant and conversational AI interaction data**
- `datasets_by_type/10_chatbot_conversational/`
- **Variables**: 13 (intents, sentiment, latency, escalation)
- **Sample sizes**: ~400, ~1,600, ~8,000 conversation turns
- **Highlights**: Dialog systems research, turn-by-turn analysis
- [ Detailed Documentation](docs/chatbot_conversational_README.md)

### Physiological & Cognitive Methods

#### 4. Eye Tracking
**Eye tracking across areas of interest**
- `datasets_by_type/04_eye_tracking/`
- **Variables**: 15 (fixations, dwell time, TTFF, pupil diameter)
- **Sample sizes**: 50, 200, 1,000 AOI observations
- **Highlights**: Visual attention analysis, cognitive load assessment
- [ Detailed Documentation](docs/eye_tracking_README.md)

#### 5. Physiological Data (EEG/GSR)
**Psychophysiological measures during engagement**
- `datasets_by_type/05_physiological/`
- **Variables**: 17 (EEG, GSR, heart rate, engagement)
- **Sample sizes**: 50, 200, 1,000 measurements
- **Highlights**: Multi-modal physiological measurement
- [ Detailed Documentation](docs/physiological_README.md)

#### 12. Cognitive Task Data
**Experimental cognitive tasks with RT/accuracy**
- `datasets_by_type/12_cognitive_task/`
- **Variables**: 14 (conditions, RT, accuracy, demographics)
- **Sample sizes**: ~2,400, ~9,600, ~48,000 trials
- **Highlights**: Stroop-like tasks, cognitive modeling data
- [ Detailed Documentation](docs/cognitive_task_README.md)

### Information Architecture Methods

#### 6. Card Sorting
**Information architecture categorization study**
- `datasets_by_type/06_card_sorting/`
- **Variables**: 10 (correctness, confidence, path length)
- **Sample sizes**: 50, 200, 1,000 categorizations
- **Highlights**: IA structure optimization, design comparison
- [ Detailed Documentation](docs/card_sorting_README.md)

### Experimental & A/B Testing Methods

#### 7. A/B Testing
**A/B test with conversion and engagement metrics**
- `datasets_by_type/07_ab_testing/`
- **Variables**: 10 (conversion, revenue, bounce rate)
- **Sample sizes**: 50, 200, 1,000 users
- **Highlights**: Hypothesis testing, business impact analysis
- [ Detailed Documentation](docs/ab_testing_README.md)

### Conversion & Retention Methods

#### 8. Funnel / Retention Data
**Multi-step conversion funnel with drop-off tracking**
- `datasets_by_type/08_funnel_retention/`
- **Variables**: 10 (step completions, retention, exit points)
- **Sample sizes**: 50, 200, 1,000 users
- **Highlights**: Funnel analysis, conversion optimization
- [ Detailed Documentation](docs/funnel_retention_README.md)

#### 16. Feature Adoption
**Feature usage and retention time series**
- `datasets_by_type/16_feature_adoption/`
- **Variables**: 14 (usage, retention days, engagement)
- **Sample sizes**: 50, 200, 1,000 users
- **Highlights**: Adoption curves, cohort retention analysis
- [ Detailed Documentation](docs/feature_adoption_README.md)

### Gaming & Spatial Interaction

#### 13. Game Analytics
**Player telemetry and engagement in games**
- `datasets_by_type/13_game_analytics/`
- **Variables**: 13 (score, XP, deaths, engagement)
- **Sample sizes**: 50, 200, 1,000 sessions
- **Highlights**: Player behavior modeling, engagement metrics
- [ Detailed Documentation](docs/game_analytics_README.md)

#### 14. Spatial / XR Data
**AR/VR spatial interaction and gaze tracking**
- `datasets_by_type/14_spatial_xr/`
- **Variables**: 13 (3D coordinates, gaze, latency)
- **Sample sizes**: 50, 200, 1,000 interactions
- **Highlights**: 3D interaction patterns, spatial analytics
- [ Detailed Documentation](docs/spatial_xr_README.md)

### Comprehensive Measurement

#### 17. System UX Metrics
**Integrated telemetry combining UX and performance**
- `datasets_by_type/17_system_ux_metrics/`
- **Variables**: 15 (success, latency, satisfaction, errors)
- **Sample sizes**: 50, 200, 1,000 task completions
- **Highlights**: Holistic UX analysis, performance correlation
- [ Detailed Documentation](docs/system_ux_metrics_README.md)

---

## Repository Structure

```
datasets_ux/
│
├── README.md                          # This file
├── generate_datasets.py               # Dataset generation script
├── requirements.txt                   # Python dependencies
├── LICENSE                            # License file
│
├── datasets_by_type/                  # Main dataset directory (TYPE-FIRST)
│   ├── 01_survey_questionnaire/
│   │   ├── small/                    # n ≈ 50
│   │   ├── medium/                   # n ≈ 200
│   │   └── large/                    # n ≈ 1,000
│   ├── 02_usability_testing/
│   ├── 03_interaction_telemetry/
│   └── ... (14 more types)
│
└── docs/                              # Detailed documentation
    ├── survey_questionnaire_README.md
    ├── usability_testing_README.md
    └── ... (15 more README files)
```

---

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mohsen-rafiei/UX_datasets.git
cd UX_datasets
```

2. **Install dependencies** (optional, for regeneration)
```bash
pip install pandas numpy scipy
# or
python -m pip install -r requirements.txt
```

### Basic Usage

#### Python
```python
import pandas as pd

# Load a specific dataset
df = pd.read_csv('datasets_by_type/01_survey_questionnaire/large/survey_questionnaire_large.csv')

# Basic statistics
print(df.describe())
print(df['Task_Success'].mean())  # Task success rate
```

#### R
```r
library(readr)

# Load dataset
df <- read_csv('datasets_by_type/01_survey_questionnaire/large/survey_questionnaire_large.csv')

# Basic statistics
summary(df)
mean(df$Task_Success)  # Task success rate
```

### Regenerating Datasets

If you want to regenerate datasets with different parameters:

```bash
python generate_datasets.py
```

This will create fresh datasets in `small/`, `medium/`, and `large/` directories.

---

## Statistical Characteristics

### Distributions Used

- **Normal**: Satisfaction scores, age, physiological measures
- **Log-normal**: Duration, latencies, dwell times, load times
- **Poisson**: Counts (errors, clicks, events, queries)
- **Beta**: Likert scales, engagement scores, confidence
- **Exponential**: Error rates, rare events
- **Binomial**: Success/failure, conversions, binary outcomes
- **Zero-inflated**: Error counts with many zeros
- **Ordinal**: Cognitive load, frustration, satisfaction scales
- **Skewed**: Ceiling/floor effects in satisfaction/frustration

### Built-in Relationships

The datasets include realistic relationships among variables:

- **Performance ↔ Satisfaction**: Higher success → Higher satisfaction
- **Workload ↔ Stress**: Higher cognitive load → Higher frustration
- **Time ↔ Accuracy**: Tradeoffs between speed and correctness
- **Device Effects**: Different patterns across device types
- **Design Effects**: Experimental conditions show differences
- **Physiological Correlations**: Cross-measure relationships in psychophysiological data
- **Accessibility Gaps**: Performance differences for users with disabilities
- **Adoption Curves**: Feature usage increases over time
- **Retention Decay**: Exponential decline in user activity

### Data Quality Features

- **Missing Values**: ~2-3% randomly distributed
- **Outliers**: ~2% of observations with extreme values
- **Noise**: Small random variation in continuous measures
- **Conditional Missingness**: Some fields only present when relevant
- **Realistic Artifacts**: True-to-life data quality issues

---

## Suggested Analyses

### Descriptive Analyses
- Summary statistics and data visualization
- Distribution analysis
- Exploratory data analysis (EDA)

### Inferential Analyses
- t-tests and ANOVAs
- Chi-square tests
- Regression models (linear, logistic, Cox survival)
- Mixed-effects models
- Correlation analysis

### Multivariate Analyses
- Factor analysis
- Principal Component Analysis (PCA)
- Clustering and segmentation
- Classification

### Specialized UX Analyses
- **SUS Score Calculation**: System usability scale computation
- **Task Success Rates**: Completion percentage analysis
- **Funnel Conversion**: Step-by-step drop-off analysis
- **Retention Cohorts**: User retention over time
- **A/B Test Significance**: Hypothesis testing
- **Cognitive Load Assessment**: Psychophysiological analysis
- **Accessibility Gap Analysis**: Performance differences
- **Attention Heatmaps**: Eye tracking visualization
- **Sentiment Analysis**: Text sentiment on diary data

---

## Course Integration Ideas

### HCI / UX Courses
- **Introduction to UX Research**: Use survey and usability testing datasets
- **Human-Computer Interaction**: Eye tracking and cognitive task data
- **Web Analytics**: Interaction telemetry and search logs
- **Game Design**: Game analytics and player telemetry

### Statistics / Data Science Courses
- **Introductory Statistics**: Small datasets for basic analyses
- **Intermediate Statistics**: Medium datasets for more complex models
- **Advanced Statistics**: Large datasets for multivariate methods
- **Data Cleaning**: Practice with missing values and outliers

### Psychology / Cognitive Science Courses
- **Cognitive Psychology**: Cognitive task and experimental data
- **Psychophysiology**: Physiological and eye tracking data
- **Behavioral Science**: Survey and diary data with sentiment

---

## Documentation

Each dataset type includes detailed documentation in the `docs/` directory:

- Study design description
- Variable explanations
- Statistical distributions
- Suggested analyses
- Data quality notes

See the individual README files for comprehensive details:
- `docs/survey_questionnaire_README.md`
- `docs/usability_testing_README.md`
- `docs/interaction_telemetry_README.md`
- ... (and 14 more)

---

## Citation

If you use these datasets in teaching or research, please cite:

```bibtex
@dataset{rafiei2025uxdatasets,
  author = {Rafiei, Mohsen},
  title = {UX Datasets Collection: Multi-method UX Research Datasets for Teaching HCI and Cognitive Psychology},
  year = {2025},
  url = {https://github.com/mohsen-rafiei/UX_datasets},
  note = {Educational dataset collection}
}
```

Or in APA format:
```
Rafiei, M. (2025). UX Datasets Collection: Multi-method UX research datasets for 
teaching HCI and cognitive psychology [Dataset]. GitHub. 
https://github.com/mohsen-rafiei/UX_datasets
```

---

## License

This dataset collection is provided for **educational and research purposes**.

-  **Free to use** for teaching and research
-  **Free to modify** and regenerate
-  **Attribution required** when using in publications or teaching materials
-  **Not for commercial use** without permission

---

## Author

**Mohsen Rafiei, Ph.D.**

Created this collection to support education in UX research, HCI, and cognitive psychology.

- **Lab Website**: [Perceptual User Experience Lab (PUX)](https://www.puxlab.com/)
- For inquiries about use in educational programs
- Questions about dataset generation
- Collaboration opportunities

---

## Acknowledgments

This dataset collection incorporates:
- **Validated UX Instruments**: SUS, UEQ, NASA-TLX scales
- **Common Research Methodologies**: Eye tracking, surveys, telemetry, experimental tasks
- **Realistic Statistical Properties**: Distributions and correlations based on research literature
- **Educational Design**: Multiple sizes and types to support progressive learning

---

## Issues & Contributions

Found an issue or want to contribute?

- Open an issue on GitHub
- Suggest improvements or additional datasets
- Share how you're using these datasets in your teaching or research

---

## Dataset Quick Reference

| Dataset | Type Directory | Small | Medium | Large |
|---------|---------------|-------|--------|-------|
| Survey & Questionnaire | `01_survey_questionnaire` | 50 | 200 | 1,000 |
| Usability Testing | `02_usability_testing` | 50 | 200 | 1,000 |
| Interaction Telemetry | `03_interaction_telemetry` | ~500 | ~2,000 | ~10,000 |
| Eye Tracking | `04_eye_tracking` | 50 | 200 | 1,000 |
| Physiological | `05_physiological` | 50 | 200 | 1,000 |
| Card Sorting | `06_card_sorting` | 50 | 200 | 1,000 |
| A/B Testing | `07_ab_testing` | 50 | 200 | 1,000 |
| Funnel Retention | `08_funnel_retention` | 50 | 200 | 1,000 |
| Search/IA | `09_search_ia` | ~500 | ~2,000 | ~10,000 |
| Chatbot | `10_chatbot_conversational` | ~400 | ~1,600 | ~8,000 |
| Accessibility | `11_accessibility` | 50 | 200 | 1,000 |
| Cognitive Task | `12_cognitive_task` | ~2,400 | ~9,600 | ~48,000 |
| Game Analytics | `13_game_analytics` | 50 | 200 | 1,000 |
| Spatial/XR | `14_spatial_xr` | 50 | 200 | 1,000 |
| Diary Qualitative | `15_diary_qualitative` | 50 | 200 | 1,000 |
| Feature Adoption | `16_feature_adoption` | 50 | 200 | 1,000 |
| System UX Metrics | `17_system_ux_metrics` | 50 | 200 | 1,000 |

**Total**: 51 CSV files (17 types × 3 sizes)

---

## Important Note

These are **synthetically generated datasets** designed to simulate real-world UX research data for educational purposes. While they mimic realistic data characteristics:
-  Suitable for **teaching statistical methods**
-  Useful for **learning data analysis**
-  Helpful for **prototyping analysis workflows**
-  **Not real research data**
-  **Should not be used** to draw conclusions about actual products or systems

---

## Star History

If you find this useful for your teaching or research, please consider giving it a  on GitHub!

---

**Last Updated**: 2025  
**Version**: 1.0.0  
**Status**: Active Development

Made with  for the UX research and HCI education community

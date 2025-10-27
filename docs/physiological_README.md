# Dataset 5: EEG / GSR / Physiological Data

## Study Design
Physiological measurement study during gameplay/engagement scenarios across different conditions.

## Sample Sizes
- **Small**: 50 measurements (~3 participants)
- **Medium**: 200 measurements (~10 participants)
- **Large**: 1,000 measurements (~50 participants)

## Variables
- `Participant_ID`: Unique identifier
- `Condition`: Game_A, Game_B, or Rest condition
- `Task_ID`: Specific task within condition
- `EEG_Alpha`: Alpha brain wave activity
- `EEG_Beta`: Beta brain wave activity
- `EEG_Theta`: Theta brain wave activity
- `EEG_Workload_Index`: Derived cognitive workload metric
- `EEG_Engagement_Index`: Derived engagement metric
- `GSR_Peak_Count`: Galvanic skin response peaks (Poisson)
- `GSR_Peak_Amplitude`: Amplitude of GSR response (log-normal)
- `Heart_Rate_BPM`: Heart rate (50-120 bpm, normal)
- `Valence`: Emotional valence (dimensionless)
- `Arousal`: Emotional arousal (dimensionless)
- `Pupil_Diameter_mm`: Pupil size measurement
- `Blink_Rate_per_min`: Eye blinks per minute
- `Timestamp`: Measurement time
- `Event_Label`: What was happening (Action, Cutscene, Menu, Loading, Combat)

## Statistical Distributions
- EEG measures: Normal distributions
- GSR: Poisson (counts) and log-normal (amplitude)
- Heart rate: Normal (70 ± 12 bpm)
- Workload metrics: Normal with positive correlations

## Relationships
- Workload ↔ Heart rate (positive correlation)
- Workload ↔ GSR amplitude (positive correlation)
- Workload ↔ EEG engagement (positive correlation)
- High workload → Increased pupil dilation
- Condition (Game vs. Rest) → Different physiological profiles

## Suggested Analyses
1. **Condition Comparisons**: ANOVA for physiological measures
2. **Correlation Analysis**: Inter-measure relationships
3. **Factor Analysis**: Dimensionality of psychophysiological measures
4. **Time Series**: Physiological responses over time
5. **Event-related Analysis**: Responses to specific events
6. **Classification**: Predicting condition from physiology

## Data Quality Notes
- ~3% missing values
- Strong correlations between measures built in
- Suitable for multivariate analyses


# Dataset 4: Eye Tracking Data

## Study Design
Eye tracking study on landing page attention patterns across different design conditions.

## Sample Sizes
- **Small**: 50 AOI-level observations
- **Medium**: 200 AOI-level observations
- **Large**: 1,000 AOI-level observations

## Variables
- `Participant_ID`: Unique participant identifier
- `Stimulus_ID`: Which design variant was shown
- `AOI_Label`: Area of Interest (Header, Navigation, Content, CTA_Button, Footer, Sidebar)
- `Fixation_Count`: Number of fixations on AOI
- `Fixation_Duration_ms`: Duration of individual fixations (log-normal)
- `Dwell_Time_ms`: Total time spent on AOI
- `Time_to_First_Fixation_ms`: TTFF - how quickly user looked at AOI
- `First_Fixation_Duration_ms`: Duration of first fixation
- `Saccade_Count`: Number of saccadic eye movements
- `Gaze_Path_Length_px`: Total distance traveled (pixels)
- `Average_Pupil_Diameter_mm`: Pupil size (2-6mm, normal)
- `Blink_Rate_per_s`: Blinks per second
- `Entry_Time_s`, `Exit_Time_s`: When user entered/exited AOI
- `Condition`: Design variant (Design_A, Design_B, Control)

## Statistical Distributions
- Fixation/Dwell times: Log-normal
- Pupil diameter: Normal (2-6mm)
- Fixation count: Poisson
- TTFF: Right-skewed, log-normal

## Relationships
- Shorter TTFF ↔ Higher task success
- More fixations ↔ Longer dwell time
- Pupil diameter ↔ Cognitive load/arousal
- Design condition → Different attention patterns

## Suggested Analyses
1. **Attention Heatmaps**: Visual attention distribution
2. **AOI Comparison**: Fixation patterns across areas
3. **Condition Effects**: Design variant impact on attention
4. **Scannpath Analysis**: Order and sequence of fixations
5. **Pupil Dilation**: Cognitive load assessment
6. **TTFF Analysis**: Time to notice key elements

## Data Quality Notes
- ~2% missing values
- TTFF inversely correlated with success
- Suitable for ANOVA/regression with condition factors


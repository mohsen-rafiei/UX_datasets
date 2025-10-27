# Dataset 1: Survey & Questionnaire Data

## Study Design
Post-task usability survey for a mobile application, combining multiple validated UX measurement instruments.

## Sample Sizes
- **Small**: 50 participants
- **Medium**: 200 participants  
- **Large**: 1,000 participants

## Variables

### Demographics
- `Participant_ID`: Unique participant identifier
- `Age`: Participant age (18-75)
- `Gender`: Gender identity (M, F, Other, Prefer not to say)
- `Device_Type`: iOS, Android, or Desktop
- `Experience_Level`: Beginner, Intermediate, or Advanced

### Task Performance
- `Task_Success`: Binary (0=failed, 1=succeeded)
- `Completion_Time_s`: Task completion time in seconds (log-normal distribution)

### SUS (System Usability Scale)
- `SUS_Q1` to `SUS_Q10`: Individual SUS items (1-5 scale)
- 10-question Likert scale measuring perceived usability

### UEQ (User Experience Questionnaire)
- `UEQ_Attractiveness`: Visual appeal dimension (1-7)
- `UEQ_Efficiency`: Performance efficiency (1-7)
- `UEQ_Perspicuity`: Ease of understanding (1-7)

### NASA-TLX (Task Load Index)
- `NASA_TLX_Mental`: Mental demand (0-21)
- `NASA_TLX_Temporal`: Time pressure (0-21)
- `NASA_TLX_Frustration`: Frustration level (0-21)

### Additional Metrics
- `Trust_Score`: User trust in the system (1-7)
- `Ease_of_Use`: Perceived ease of use (1-7)
- `Willingness_to_Reuse`: Likelihood to use again (1-7)
- `Attention_Check_Passed`: Quality control check (0=no, 1=yes)

## Statistical Distributions
- SUS/UEQ scales: Likert-type, slightly skewed toward positive
- Completion time: Log-normal
- NASA-TLX: Exponential/log-normal with ceiling effects
- Trust/Usability: Beta distribution mapped to 1-7 scale

## Relationships
- Higher task success → Higher trust scores and willingness to reuse
- Higher task success → Lower NASA-TLX frustration scores
- Longer completion time (failures) → Higher mental demand and frustration

## Suggested Analyses
1. **SUS Score Calculation**: Composite usability score from Q1-Q10
2. **Correlational Analysis**: Task success vs. subjective metrics
3. **Group Comparisons**: Device type or experience level effects
4. **Regression**: Predicting satisfaction from performance metrics
5. **Factor Analysis**: Inter-relationships among UEQ dimensions

## Data Quality Notes
- ~2% missing values injected to simulate real survey data
- Some outliers in completion time (~2%)
- Attention check allows identification of poor-quality responses


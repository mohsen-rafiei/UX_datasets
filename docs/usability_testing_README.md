# Dataset 2: Usability Testing Data

## Study Design
Controlled usability test for web checkout process, with multiple tasks and direct observation.

## Sample Sizes
- **Small**: 50 task observations
- **Medium**: 200 task observations
- **Large**: 1,000 task observations

## Variables
- `Participant_ID`: Unique participant identifier
- `Task_Name`: Specific task (Checkout, Product Search, Account Setup, Payment)
- `Task_Success`: Binary completion (0/1)
- `Completion_Time_s`: Time to complete task (right-skewed, log-normal)
- `Error_Count`: Number of errors (Poisson distribution)
- `Error_Type`: Type of error (Validation, Navigation, Input, System, None)
- `Frustration_Level`: Self-reported frustration (1-5 Likert)
- `Help_Requested`: Whether user requested assistance (0/1)
- `Moderator`: Which moderator conducted the test
- `Device_Type`: Desktop, Tablet, or Mobile
- `Satisfaction_Score`: Post-task satisfaction (3-7)

## Statistical Distributions
- Completion time: Right-skewed, log-normal
- Error count: Poisson (λ=0.5-2.5 depending on success)
- Satisfaction: Near-ceiling effect (positive skew)

## Relationships
- Task failure → Longer completion time (~1.5x)
- Task failure → Higher error count
- Error count → Higher frustration level
- Task success → Lower frustration

## Suggested Analyses
1. **Task Success Rates**: Percentage successful for each task
2. **Error Analysis**: Most common error types by task
3. **Time Comparisons**: Between successful vs. failed tasks
4. **Device Effects**: Performance differences across device types
5. **Moderator Effects**: Inter-rater reliability analysis
6. **Cox Regression**: Time-to-success modeling

## Data Quality Notes
- ~2.5% missing values
- Frustration scores correlated with performance


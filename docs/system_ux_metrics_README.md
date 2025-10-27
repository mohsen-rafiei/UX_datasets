# Dataset 17: System-Level UX Metrics

## Study Design
Continuous telemetry tracking across multiple product features with system performance integration.

## Sample Sizes
- **Small**: 50 task completions
- **Medium**: 200 task completions
- **Large**: 1,000 task completions

## Variables
- `User_ID`: Unique identifier
- `Session_ID`: Session identifier
- `Task_Name`: Login, Upload, Search, Checkout, Settings
- `Task_Success`: Binary completion (0/1, ~80% success)
- `Time_to_Complete_s`: Task duration (log-normal)
- `Error_Rate_%`: Error rate (0-50%, exponential with success effect)
- `Satisfaction_Score`: User satisfaction (1-7, normal)
- `Cognitive_Load_Score`: Mental load (1-5, ordinal)
- `System_Latency_ms`: Backend latency (log-normal)
- `Click_Error_Rate`: UI interaction errors (0-20%, exponential)
- `Average_Path_Length`: Number of clicks to complete
- `Help_Accessed`: Whether help was used (binary, ~25%)
- `Post_Task_Confidence`: User confidence (3-5, ordinal, positively skewed)
- `Device_OS`: Windows, MacOS, iOS, or Android
- `Timestamp`: When task was completed

## Statistical Distributions
- Error rates: Exponential, varies by success
- Satisfaction: Normal distribution
- Latency: Log-normal
- Cognitive load: Ordinal with correlations
- Confidence: Positively skewed ordinal

## Relationships
- Task success ↔ Lower error rate
- System latency ↔ Lower satisfaction
- Error rate ↔ Higher cognitive load
- Error rate ↔ Lower confidence
- Task success ↔ Lower latency (indirect)
- Satisfaction belong with lower error rate and latency

## Suggested Analyses
1. **Performance Monitoring**: System health dashboards
2. **Task Analysis**: Success rates by task type
3. **Correlation Analysis**: Metrics interrelationships
4. **Predictive Modeling**: Factors affecting satisfaction
5. **Goal Tech**: Device/OS differences
6. **Error Analysis**: Common error patterns

## Data Quality Notes
- ~2% missing values
- Strong correlations between UX and performance metrics
- Suitable for holistic UX analysis


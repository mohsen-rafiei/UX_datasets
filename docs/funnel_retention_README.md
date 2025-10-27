# Dataset 8: Funnel / Retention Data

## Study Design
Signup funnel analysis with step-by-step completion tracking and retention metrics.

## Sample Sizes
- **Small**: 50 users
- **Medium**: 200 users
- **Large**: 1,000 users

## Variables
- `User_ID`: Unique identifier
- `Session_ID`: Session identifier
- `Referral_Source`: Organic, Paid, Referral, Direct, Social
- `Device_Type`: Desktop, Mobile, or Tablet
- `Region`: North, South, East, West
- `Step_1_Visited`: Visited first step (binary)
- `Step_2_Clicked`: Clicked to second step (binary, conditional on Step_1)
- `Step_3_Submitted`: Submitted form (binary, conditional on Step_2)
- `Step_4_Confirmed`: Confirmed submission (binary, conditional on Step_3)
- `Completion`: Successfully completed funnel (binary)
- `Time_Between_Steps_s`: Time between steps (log-normal)
- `Exit_Step`: Which step user exited (1-4, 0 if completed)

## Statistical Distributions
- Step completion: Binomial with cascading probabilities
- Drop-off probabilities: ~5% (Step 1), ~25% (Step 2), ~50% (Step 3), ~70% (Step 4)
- Time between steps: Right-skewed, log-normal

## Relationships
- Completion probability decreases at each step
- Longer time between steps → Higher drop-off risk
- Source effects: Paid traffic has different patterns
- Device → Different completion rates

## Suggested Analyses
1. **Funnel Visualization**: Drop-off at each step
2. **Cohort Analysis**: User behavior by source
3. **Survival Analysis**: Time to drop-off
4. **Conversion Rate**: Overall and by segment
5. **Causality**: Which factors predict completion
6. **Optimization**: Where to focus improvement efforts

## Data Quality Notes
- ~2% missing values
- Conditional binary variables (steps cascade)


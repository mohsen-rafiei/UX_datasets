# Dataset 16: Feature Adoption / Time Series Data

## Study Design
Feature adoption tracking after product launch with retention metrics over 30-day period.

## Sample Sizes
- **Small**: 50 users
- **Medium**: 200 users
- **Large**: 1,000 users

## Variables
- `User_ID`: Unique identifier
- `Cohort_Date`: When user signed up
- `Feature_Name`: Search, Notifications, Profile_Edit, Export_Data, Dark_Mode
- `Feature_Used`: Whether user tried the feature (binary)
- `Days_Since_Launch`: Days from feature release (0-60)
- `Usage_Count`: Number of times feature used (Poisson)
- `Retention_Day_1`: Active on Day 1 (binary)
- `Retention_Day_7`: Active on Day 7 (binary)
- `Retention_Day_30`: Active on Day 30 (binary)
- `Session_Length_s`: Average session length
- `Engagement_Score`: Overall engagement (Beta distribution, 0-100)
- `Subscription_Status`: Premium vs. free (binary, ~35% premium)
- `Region`: US, EU, ASIA, Other
- `Device_Type`: Mobile, Desktop, or Tablet

## Statistical Distributions
- Retention: Binary, decays exponentially (70% → 50% → 40%)
- Usage count: Poisson, conditional on adoption
- Engagement: Beta distribution
- Feature usage: Increases over time (adoption curve)

## Relationships
- Days since launch → Higher adoption rate
- Feature usage → Higher engagement score
- Retention Day 1 → Conditional retention Day 7
- Retention Day 7 → Conditional retention Day 30
- Subscribed users → Higher engagement

## Suggested Analyses
1. **Adoption Curves**: Feature usage over time
2. **Retention Analysis**: Cohort retention rates
3. **Survival Analysis**: Time to feature adoption
4. **Engagement**: Usage vs. engagement correlation
5. **Segmentation**: Usage by region/device
6. **Growth Metrics**: DAU, MAU, stickiness

## Data Quality Notes
- ~2% missing values
- Time series structure with cohort effects
- Suitable for growth analysis


# Dataset 7: A/B Testing Data

## Study Design
A/B test comparing two homepage designs (Variant A vs. B) with conversion and engagement metrics.

## Sample Sizes
- **Small**: 50 users
- **Medium**: 200 users
- **Large**: 1,000 users

## Variables
- `User_ID`: Unique user identifier
- `Variant`: A or B (randomly assigned)
- `Timestamp`: When user visited
- `Device_Type`: Desktop, Mobile, or Tablet
- `Country`: User location (US, UK, DE, FR, CA)
- `Conversion`: Binary conversion event (0/1)
- `Click_Count`: Number of clicks during session
- `Page_Load_Time_ms`: Page load performance
- `Session_Length_s`: Total session duration
- `Revenue_USD`: Revenue generated (heavy-tailed, 0 if no conversion)
- `Bounce_Rate_%`: Bounce rate percentage

## Statistical Distributions
- Conversion: Binary (Variant B has higher rate: 8% vs. 5%)
- Revenue: Exponential distribution, only for converters
- Page load time: Log-normal
- Click count: Poisson with conversion boost

## Relationships
- Conversion → Higher click count
- Conversion → Longer session length
- Conversion → Revenue generation
- Conversion → Lower bounce rate
- Variant B → Better performance metrics

## Suggested Analyses
1. **Conversion Rate Comparison**: Chi-square or proportion test
2. **Revenue Analysis**: Mean differences across variants
3. **Segmentation**: Device or country effects
4. **Survival Analysis**: Time to conversion
5. **Statistical Significance**: p-value calculations
6. **Toggle Work**: Effect size and business impact

## Data Quality Notes
- ~2% missing values
- Built-in positive effect for Variant B
- Suitable for hypothesis testing


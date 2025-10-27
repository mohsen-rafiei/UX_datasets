# Dataset 3: Interaction / Telemetry Log Data

## Study Design
Website interaction tracking through automated event logging (clickstream data).

## Sample Sizes
- **Small**: ~500 events (50 users × ~10 events)
- **Medium**: ~2,000 events (200 users × ~10 events)
- **Large**: ~10,000 events (1,000 users × ~10 events)

## Variables
- `User_ID`: Unique user identifier
- `Session_ID`: Session identifier
- `Timestamp`: When the event occurred
- `Event_Type`: Click, scroll, page_view, form_submit, hover
- `Event_Target`: Which element was targeted
- `Click_X`, `Click_Y`: Mouse coordinates (if click/hover)
- `Scroll_Depth_%`: How far user scrolled (0-100%)
- `Dwell_Time_ms`: Time hovering over element (log-normal)
- `Time_on_Page_s`: Total time on page
- `Session_Length_s`: Duration of entire session
- `Device_Type`: Desktop, Mobile, or Tablet
- `Browser`: Chrome, Safari, Firefox, or Edge
- `Network_Speed_Mbps`: Connection speed
- `Error_Code`: System errors (404, 500, Timeout, None)
- `Conversion`: Binary conversion event (0/1)

## Statistical Distributions
- Dwell time: Log-normal
- Network speed: Log-normal
- Conversion: Binary (rare event ~5%)
- Session metrics: Heavily right-skewed

## Relationships
- Higher scroll depth → Longer time on page
- Conversion → Longer session length
- Conversion → More clicks
- Errors → Lower conversion probability

## Suggested Analyses
1. **User Journey Mapping**: Common paths through the site
2. **Click Heatmaps**: Spatial distribution of clicks
3. **Conversion Funnel**: Drop-off at each step
4. **Time Series**: Session patterns over time
5. **Segmentation**: High vs. low engagement users
6. **A/B Testing**: Event-level comparisons

## Data Quality Notes
- ~3% missing values
- Conditional missingness (e.g., Click_X only for click events)
- Large sample sizes suitable for segmentation analysis


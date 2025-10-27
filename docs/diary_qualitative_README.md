# Dataset 15: Diary / Mixed-Method Qualitative Data

## Study Design
Remote diary study with user-generated text responses and sentiment analysis.

## Sample Sizes
- **Small**: 50 diary entries
- **Medium**: 200 diary entries
- **Large**: 1,000 diary entries

## Variables
- `Participant_ID`: Unique identifier
- `Entry_Date`: When entry was made
- `Prompt`: Morning_Routine, Afternoon_Activity, Evening_Reflection, Weekly_Summary
- `Response_Text`: User's text response
- `Emotion_Label`: Positive, Neutral, Negative, Mixed
- `Sentiment_Score`: Sentiment analysis score (-1 to 1)
- `Word_Count`: Length of response (log-normal)
- `Theme_Code`: Usability, Aesthetics, Performance, Value, Emotional_Response
- `Device_Used`: Mobile, Desktop, or Tablet
- `Environment`: Home, Office, Transit, Other
- `Follow_Up_Flag`: Whether entry requires follow-up (binary)

## Statistical Distributions
- Sentiment: Normal, centered near 0, varies by emotion label
- Word count: Log-normal
- Follow-up: Binary (~20%)

## Relationships
- Emotion label ↔ Sentiment score
- Longer entries (word count) → More detailed insights
- Negative emotion → Higher follow-up probability
- Environment → Different themes (Transit→Performance)

## Suggested Analyses
1. **Sentiment Analysis**: Emotion evolution over time
2. **Thematic Analysis**: Most common themes
3. **Qualitative Coding**: Text content analysis
4. **Temporal Patterns**: Time-of-day effects
5. **Environment Effects**: Context impact on feedback
6. **Mixed Methods**: Combining qualitative text with quantitative scores

## Data Quality Notes
- ~2.5% missing values
- Realistic sample responses included
- Suitable for qualitative/quantitative hybrid analysis


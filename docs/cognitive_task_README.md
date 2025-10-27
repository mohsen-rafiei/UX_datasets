# Dataset 12: Experimental Cognitive Task Data

## Study Design
Laboratory cognitive task (Stroop or visual search) with reaction time and accuracy measures.

## Sample Sizes
- **Small**: ~2,400 trials (~50 participants × ~48 trials)
- **Medium**: ~9,600 trials (~200 participants × ~48 trials)
- **Large**: ~48,000 trials (~1,000 participants × ~48 trials)

## Variables
- `Participant_ID`: Unique identifier
- `Trial_ID`: Trial number (1-48 per participant)
- `Condition`: Congruent, Incongruent, or Neutral
- `Stimulus_Word`: Word displayed
- `Stimulus_Color`: Color of word
- `Response_Key`: Response key pressed (e.g., 'Space')
- `Correct_Response`: Expected response (0/1)
- `Reaction_Time_ms`: Response time (log-normal)
- `Accuracy`: Correct/incorrect (binary)
- `Trial_Order`: Position in sequence
- `Feedback`: Feedback provided
- `Age`: Participant age (18-65)
- `Gender`: Participant gender
- `Session_Number`: Which session (1-3)

## Statistical Distributions
- Reaction time: Log-normal
- Accuracy: Binary, varies by condition
- Condition effects: Congruent > Neutral > Incongruent

## Relationships
- Condition → Accuracy (congruent easiest, 95%)
- Condition → Reaction time (incongruent slowest, 6.5s mean)
- Higher accuracy → Shorter RT (negative correlation)
- Session number → Learning effects
- Age → Performance decline

## Suggested Analyses
1. **Condition Effects**: ANOVA on RT and accuracy
2. **Stroop Effect**: Congruent vs. incongruent comparison
3. **Practice Effects**: Session/order effects
4. **Individual Differences**: Age, gender effects
5. **RT Distribution Analysis**: Ex-Gaussian fitting
6. **Error Analysis**: Commission vs. omission errors

## Data Quality Notes
- ~2% missing values
- Strong experimental effect (congruency)
- Suitable for cognitive modeling


# Dataset 11: Accessibility Testing Data

## Study Design
Form usability testing with assistive technologies for users with various disabilities.

## Sample Sizes
- **Small**: 50 participants
- **Medium**: 200 participants
- **Large**: 1,000 participants

## Variables
- `Participant_ID`: Unique identifier
- `Disability_Type`: Visual, Motor, Cognitive, or Hearing
- `Assistive_Tech`: Screen_Reader, Voice_Control, Keyboard_Only, Magnifier, None
- `Task_Name`: Form_Submit, Navigation, Checkout, Search
- `Task_Success`: Binary completion (0/1)
- `Completion_Time_s`: Time to complete (log-normal, higher for disabilities)
- `Error_Count`: Number of errors (zero-inflated Poisson)
- `Error_Type`: Navigation, Labeling, Focus, Compatibility, None
- `Keyboard_Presses`: Total keyboard interactions (Poisson)
- `Cognitive_Load_Score`: Self-reported cognitive load (1-5)
- `Satisfaction_Score`: Post-task satisfaction (4-7, skewed high)
- `WCAG_Issue_Count`: Number of WCAG compliance issues (Poisson)

## Statistical Distributions
- Error count: Zero-inflated Poisson
- Satisfaction: Positively skewed (ceiling effect)
- Completion time: Log-normal, with disability multiplier
- Cognitive load: Ordinal, positively correlated with errors

## Relationships
- Disability → Lower task success (~35% failure vs. ~15%)
- Disability → Longer completion time (~1.5x)
- Error count → Lower satisfaction
- Error count → Higher cognitive load
- Assistive tech matched to disability (e.g., Visual → Screen Reader)

## Suggested Analyses
1. **Accessibility Gap Analysis**: Performance by disability type
2. **Assistive Tech Effectiveness**: Tech impact on success
3. **Error Analysis**: Most problematic error types
4. **Comparative Usability**: Accessibility barriers impact
5. **WCAG Compliance**: Issue count analysis
6. **Inclusive Design**: Recommendations for improvement

## Data Quality Notes
- ~2% missing values
- Built-in performance gaps for accessibility
- Suitable for inclusive design research


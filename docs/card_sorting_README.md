# Dataset 6: Card Sorting / Tree Testing Data

## Study Design
Card sorting study for university website redesign, testing current vs. proposed information architecture.

## Sample Sizes
- **Small**: 50 card categorizations
- **Medium**: 200 card categorizations
- **Large**: 1,000 card categorizations

## Variables
- `Participant_ID`: Unique identifier
- `Card_ID`: Individual card identifier
- `Card_Label`: Card name (Home, About, Contact, Courses, Faculty, Research, Admissions, Campus)
- `Assigned_Category`: Category participant placed card in (Main_Nav, Academics, Administration, Resources)
- `Category_Correct`: Whether categorization was "correct" (0/1)
- `Confidence_Rating`: Self-reported confidence (1-5, skewed high)
- `Task_Success`: Overall task success (0/1)
- `Path_Length`: Number of steps taken (Poisson)
- `Time_on_Task_s`: Time to complete task
- `Design_Version`: Current or Proposed design

## Statistical Distributions
- Path length: Poisson
- Confidence: Positively skewed (toward high confidence)
- Time: Log-normal
- Correctness: Binary with design effects

## Relationships
- Proposed design → Higher correctness rates
- Higher confidence ↔ Correct categorization
- Longer path length → Longer time
- Design version → Different accuracy patterns

## Suggested Analyses
1. **Similarity Matrix**: Card grouping patterns
2. **Consensus Analysis**: Agreement between commanders
3. **Design Comparison**: Current vs. proposed accuracy
4. **Category Analysis**: Most problematic cards
5. **Confidence Calibration**: Confidence vs. accuracy
6. **Hierarchical Clustering**: Optimal IA structure

## Data Quality Notes
- ~2% missing values
- Built-in design effect (proposed > current)


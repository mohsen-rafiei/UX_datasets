# Dataset 9: Search / Information Architecture Log Data

## Study Design
Internal product search engine usage log data with query reformulation and click behavior.

## Sample Sizes
- **Small**: ~500 search queries (100 users)
- **Medium**: ~2,000 search queries (400 users)
- **Large**: ~10,000 search queries (2,000 users)

## Variables
- `User_ID`: Unique user identifier
- `Session_ID`: Search session
- `Query_Text`: Actual search query
- `Query_Length`: Number of words in query
- `Search_Latency_ms`: Time to return results (log-normal)
- `Results_Displayed`: Number of results shown (10, 20, 50)
- `Clicked_Result_Rank`: Position of clicked result (1-5, or None)
- `Click_Success`: Binary (1 if clicked within top 3)
- `Reformulation_Count`: Number of query refinements (Poisson)
- `Task_Success`: Overall task completion (binary)
- `Time_to_Completion_s`: Time to find desired information
- `Dwell_Time_on_Result_ms`: Time on clicked result (if applicable)

## Statistical Distributions
- Query length: Poisson
- Search latency: Log-normal
- Reformulation: Poisson (higher for failures)
- Dwell time: Log-normal (conditional on click)

## Relationships
- Task failure → More reformulations
- High result rank (top 3) → Success
- Task success → Shorter time to completion
- Click on lower rank → Longer completion time
- Reformulations → Higher task success (indirect)

## Suggested Analyses
1. **Query Analysis**: Most common queries and patterns
2. **Click-through Rate**: By result rank
3. **Reformulation Analysis**: Refinement patterns
4. **Performance Metrics**: Task success by query characteristics
5. **Search Quality**: Latency vs. success
6. **Path Analysis**: Query→Click→Reformulation sequences

## Data Quality Notes
- ~3% missing values
- Conditional missingness (dwell time only for clicks)
- Large samples suitable for text analysis


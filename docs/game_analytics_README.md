# Dataset 13: Game Analytics / Player Telemetry

## Study Design
Player behavior and engagement tracking in video game across different modes (cooperative, competitive, solo).

## Sample Sizes
- **Small**: 50 gaming sessions
- **Medium**: 200 gaming sessions
- **Large**: 1,000 gaming sessions

## Variables
- `Player_ID`: Unique player identifier
- `Session_ID`: Session identifier
- `Level_Number`: Current level (1-20)
- `Play_Duration_s`: Total play time (log-normal)
- `Actions_Per_Minute`: Input frequency (Poisson)
- `Deaths`: Number of player deaths (Poisson)
- `Score`: Points earned (lognormal with long tail)
- `XP_Gained`: Experience points (log-normal)
- `Time_in_Menu_s`: Time in menus (log-normal)
- `Item_Used`: Item consumed (Health_Potion, Mana_Potion, Grenade, Shield, None)
- `Game_Mode`: Cooperative, Competitive, or Solo
- `Engagement_Score`: Derived engagement metric (Beta distribution)
- `Session_Success`: Successfully completed session (binary, ~70%)

## Statistical Distributions
- Duration and XP: Log-normal
- Score: Normal with long tail
- Actions: Poisson
- Deaths: Poisson
- Engagement: Beta distribution

## Relationships
- Session success → Higher score and XP
- Deaths → Longer play duration (retry loops)
- Competitive mode → Higher action rate
- Engagement → Lower menu time

## Suggested Analyses
1. **Engagement Metrics**: Play patterns by mode
2. Rare Lifecycle: Player journey analysis
3. **Difficulty Progression**: Level difficulty curves
4. **Item Usage**: Consumption patterns
5. **Player Segmentation**: High vs. low engagement
6. **Retention**: Session frequency patterns

## Data Quality Notes
- ~2.5% missing values
- Suitable for player behavior modeling


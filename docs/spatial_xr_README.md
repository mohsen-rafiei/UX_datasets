# Dataset 14: Geo / XR / Spatial Interaction Data

## Study Design
Spatial interaction study in AR/VR environment with spatial targeting and cognitive load assessment.

## Sample Sizes
- **Small**: 50 interactions
- **Medium**: 200 interactions
- **Large**: 1,000 interactions

## Variables
- `User_ID`: Unique identifier
- `Device_Type`: Oculus_Quest, HTC_Vive, or AR_Glasses
- `Latitude`, `Longitude`: Geographic position
- `Session_Timestamp`: When interaction occurred
- `Gaze_Target`: What user was looking at (UI_Panel, Avatar, Object_A, etc.)
- `Interaction_Type`: Gaze, Pointer, Gesture, or Voice
- `Task_Success`: Binary success (0/1)
- `Distance_to_Target_m`: Distance to interaction target
- `Head_Orientation_XYZ`: 3D orientation as tuple string
- `Latency_ms`: System response latency (log-normal)
- `Completion_Time_s`: Task completion time
- `Cognitive_Load_Score`: Self-reported load (1-5)

## Statistical Distributions
- Position coordinates: Uniform in specified range
- Latency: Log-normal
- Completion time: Log-normal
- Cognitive load: Ordinal

## Relationships
- Device type → Different interaction patterns
- Distance to target → Longer completion time
- Higher latency → Higher cognitive load
- Task success → Lower cognitive load

## Suggested Analyses
1. **Spatial Analysis**: 3D interaction patterns
2. **Device Comparison**: VR headset performance
3. **Gaze Analysis**: Attention in 3D space
4. **Latency Impact**: Performance effects on UX
5. **Interaction Modalities**: Gaze vs. gesture effectiveness
6. **Cognitive Load**: Spatial task difficulty

## Data Quality Notes
- ~2% missing values
- 3D coordinates as formatted strings
- Suitable for spatial analytics


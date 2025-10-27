# Dataset 10: Chatbot / Conversational UX Data

## Study Design
Voice assistant or chat AI evaluation with turn-by-turn conversation tracking and intent recognition.

## Sample Sizes
- **Small**: ~400 conversation turns (50 users × ~8 turns)
- **Medium**: ~1,600 conversation turns (200 users × ~8 turns)
- **Large**: ~8,000 conversation turns (1,000 users × ~8 turns)

## Variables
- `User_ID`: Unique user identifier
- `Session_ID`: Conversation session
- `Turn_Number`: Position in conversation (1-8)
- `User_Message_Text`: User input text
- `Bot_Response_Text`: Bot response text
- `Intent_Label`: Recognized intent (order_status, return, billing, technical_support, general_info)
- `Intent_Confidence`: Confidence in intent recognition (0-1, Beta)
- `Response_Latency_ms`: Time for bot to respond (log-normal)
- `Resolution_Success`: Task successfully completed (binary)
- `User_Sentiment_Score`: Sentiment of user message (-1 to 1)
- `Escalated_to_H兴盛an`: Whether escalated (binary)
- `Task_Type`: Transactional, Informational, or Support
- `End_of_Session`: Is this the last turn (binary)

## Statistical Distributions
- Response latency: Right-skewed, log-normal
- Intent confidence: Beta distribution
- Resolution success: Binary (~70% success)
- Sentiment: Normal centered near 0
- Escalation: Binary (rare ~15%)

## Relationships
- Resolution failure → Longer latency
- Resolution failure → Higher escalation probability
- Lower confidence → More reformulations
- Negative sentiment → Escalation
- Later turns → Higher success (learning effect)

## Suggested Analyses
1. **Intent Analysis**: Distribution and accuracy by intent
2. **Success Rates**: Resolution success by intent/turn
3. **Latency Analysis**: Response time patterns
4. **Sentiment Tracking**: Sentiment evolution over conversation
5. **Escalation Triggers**: What leads to human handoff
6. **Conversation Flow**: Common conversation patterns

## Data Quality Notes
- ~2.5% missing values
- Turn-sequential structure
- Suitable for dialog systems research


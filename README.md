# Aegis - Discord Spam Detection System

## Project Overview
Aegis is a specialized spam detection system designed to combat betting promoter spam in Discord servers. The system consists of:
- An LSTM neural network trained on betting-related spam patterns
- A Discord bot interface for real-time message monitoring

## Model Training Details

### Training Data
- **Non-spam messages**: Collected from legitimate conversations in betting-related Discord servers (`data/real_messages.csv`)
- **Spam messages**: Synthetic samples generated from known spam messages promoting betting schemes (`data/spam_messages.txt`)

### Key Training Considerations
1. **Domain-Specific Focus**:
   - Model optimized for detecting betting site promotions
   - Preserves emojis/special characters (🔥, 💰, 🎯) common in gambling spam
   - Vocabulary tuned for gambling-related terminology

2. **Model Architecture**:
```python
Sequential(
    Embedding(10000, 100, input_length=50),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
)
```
3. **Performance Characteristics**:
- Typical accuracy: 92-95% on betting spam detection
- Lower effectiveness on non-gambling related spam
- Requires retraining for new spam categories

4. **Discord Bot Implementation (Aegis)**
- Features:
   - Real-time message scanning using spam_detector_model
    - Automated spam handling:
    - Delete high-confidence spam messages
    - Temporary mute for repeat offenders
    - Moderator alert channel for borderline cases
    - Whitelist system for trusted users/roles
    - Custom threshold configuration per server
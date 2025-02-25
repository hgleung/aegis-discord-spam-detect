# Aegis - Discord Spam Detection System

## Project Overview
Aegis is a specialized spam detection system designed to combat betting promoter spam in Discord servers. The system consists of:
- An LSTM neural network trained on betting-related spam patterns
- A Discord bot interface for real-time message monitoring
- Interactive learning capabilities for continuous model improvement

## Spam Detection Components

### 1. Text Processing (`clean_text`)
- **Character Substitution Handling**:
  - Normalizes common evasion tactics (e.g., "p!cks" → "picks")
  - Handles number substitutions (1 → i, 3 → e, 0 → o)
  - Processes symbol replacements (@ → a, $ → s)
- **Optimization**:
  - Skips processing for short words (≤ 2 characters)
  - Only processes words containing special characters
  - Preserves emojis and legitimate special characters

### 2. Data Management
- **Loading Data** (`load_data`):
  - Reads real messages from CSV format
  - Loads spam messages from text file
  - Combines and labels datasets (0 for real, 1 for spam)

- **Sequence Preparation** (`prepare_sequences`):
  - Tokenizes text using up to 10,000 most common words
  - Pads sequences to 50 tokens
  - Handles out-of-vocabulary words with '<OOV>' token

### 3. Model Architecture (`create_model`)
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

### 4. Model Operations
- **Prediction** (`predict_message`):
  - Processes new messages through the cleaning pipeline
  - Returns spam probability and binary classification
  - Threshold-based classification (default: 0.5)

- **Model Persistence** (`save_model_artifacts`, `load_model_artifacts`):
  - Saves/loads model weights
  - Preserves tokenizer vocabulary
  - Maintains consistent sequence parameters

### 5. Interactive Learning System
- **Feedback Processing**:
  - Collects user feedback on predictions
  - Handles false negatives through model updates
  - Saves new spam patterns for future training

- **Model Updates** (`add_spam_message`, `update_model_with_message`):
  - Balanced mini-batch training (20 messages per class)
  - Higher learning rate (0.001) for quick adaptation
  - Preserves existing knowledge while learning new patterns
  - Immediate model update verification

### 6. Usage Modes

#### Interactive Mode
```bash
python spam_detector.py
```
- Continuous message input/evaluation loop
- Real-time prediction feedback
- Model updates based on user corrections

#### Key Commands
- Enter message to check for spam
- Provide feedback (y/n) on prediction accuracy
- 'quit'/'q'/'exit' to end session

## Performance Characteristics
- Typical accuracy: 92-95% on betting spam detection
- Lower effectiveness on non-gambling related spam
- Requires retraining for new spam categories

## Discord Bot Implementation (Aegis)
- Features:
  - Real-time message scanning using spam_detector_model
  - Automated spam handling:
    - Delete high-confidence spam messages
    - Temporary mute for repeat offenders
    - Moderator alert channel for borderline cases
  - Whitelist system for trusted users/roles
  - Custom threshold configuration per server

## Model Training Details

### Training Data
- **Non-spam messages**: Collected from legitimate conversations (`data/real_messages.csv`)
- **Spam messages**: Synthetic samples from known spam patterns (`data/spam_messages.txt`)

### Training Process
1. **Initial Training**:
   - 80/20 train-test split
   - Early stopping with 3 epochs patience
   - Validation split: 0.2
   - Batch size: 32
   - Maximum epochs: 10

2. **Incremental Updates**:
   - Mini-batch size: 8
   - Training epochs: 3
   - Temporarily increased learning rate
   - Balanced sampling of existing data

## Future Improvements
- Deployment as a Discord bot
- Generalization to non-gambling related spam
- False positive handling
- Additional model architecture tuning

## Generalization Potential

### Expanding Beyond Betting Spam
The current model architecture can be adapted for different spam categories by:
- Retraining on new domain-specific datasets
- Adjusting the vocabulary size and embedding dimensions
- Fine-tuning hyperparameters for the new domain

#### Potential Application Domains
1. **Cryptocurrency Scams**
   - Token pump-and-dump schemes
   - Fake exchange promotions
   - Mining pool scams
   
2. **Phishing Attempts**
   - Fake Discord nitro offers
   - Account verification scams
   - Server raid coordination

3. **Commercial Spam**
   - Unauthorized server advertisements
   - Drop shipping promotions
   - Account selling/trading
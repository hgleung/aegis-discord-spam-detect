import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def clean_text(text):
    """
    Clean text by handling common character substitutions used to evade detection.
    Handles cases like:
    - Symbol substitutions: ! for i, @ for a, 3 for e, etc.
    - Number substitutions: 1 for i/l, 0 for o, etc.
    - Censoring characters: Preserves * and # as they indicate attempted evasion
    """
    # Common substitutions dictionary
    substitutions = {
        '!': 'i',
        '1': 'i',
        '@': 'a',
        '4': 'a',
        '3': 'e',
        '0': 'o',
        '$': 's',
        '5': 's',
        '7': 't',
        '9': 'g',
    }
    
    # Convert text to lowercase
    text = text.lower().strip()
    
    # Split into words to handle each word separately
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Skip short words or words without special characters
        if len(word) <= 2 or not any(char in substitutions or char in '*#' for char in word):
            cleaned_words.append(word)
            continue
            
        # Clean the word
        cleaned_word = word
        
        # Replace common substitutions
        for char, replacement in substitutions.items():
            cleaned_word = cleaned_word.replace(char, replacement)
        
        # Keep censoring characters as they indicate potential evasion attempts
        cleaned_words.append(cleaned_word)
    
    # Rejoin words
    return ' '.join(cleaned_words)

def load_data():
    # Load real messages
    df_real = pd.read_csv('data/real_messages.csv')
    real_messages = df_real['Content'].astype(str).apply(clean_text).tolist()
    
    # Load spam messages
    with open('data/spam_messages.txt', 'r') as f:
        spam_messages = [clean_text(line.strip()) for line in f.readlines()]
    
    # Create labels (0 for real, 1 for spam)
    real_labels = np.zeros(len(real_messages))
    spam_labels = np.ones(len(spam_messages))
    
    # Combine messages and labels
    all_messages = real_messages + spam_messages
    all_labels = np.concatenate([real_labels, spam_labels])
    
    return all_messages, all_labels

def prepare_sequences(messages, max_words=10000, max_len=50, tokenizer=None):
    # Create and fit tokenizer
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(messages)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(messages)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def create_model(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def save_model_artifacts(model, tokenizer, max_len):
    """Save the trained model, tokenizer, and parameters"""
    # Save the model
    model.save('spam_detector_model')
    
    # Save the tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the max_len parameter
    with open('params.pickle', 'wb') as handle:
        pickle.dump({'max_len': max_len}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_artifacts():
    """Load the trained model, tokenizer, and parameters"""
    # Check if all required files exist
    required_files = [
        'spam_detector_model',
        'tokenizer.pickle',
        'params.pickle'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Missing required file: {file}")
            return None, None, None
        
    try:
        # Load the model
        model = load_model('spam_detector_model')
        
        # Load the tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Load the parameters
        with open('params.pickle', 'rb') as handle:
            params = pickle.load(handle)
        
        return model, tokenizer, params['max_len']
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def predict_message(message, model, tokenizer, max_len):
    """Predict if a message is spam"""
    # Clean the message
    cleaned_message = clean_text(message)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_message])
    
    # Pad sequence
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    return prediction, prediction > 0.5

def update_spam_dataset(new_message):
    """Add a new spam message to the dataset"""
    # Clean the message
    cleaned_message = clean_text(new_message)
    
    # Append to spam_messages.txt
    with open('data/spam_messages.txt', 'a') as f:
        f.write(f"\n{cleaned_message}")

def update_model_with_message(message, model, tokenizer, max_len, training_epochs=3):
    """Update the model with a new spam message using balanced mini-batch training"""
    # Clean and prepare the new message
    cleaned_message = clean_text(message)
    
    # Load a small balanced batch of existing data
    messages, labels = load_data()
    
    # Sample a balanced subset of existing data
    real_indices = np.where(labels == 0)[0]
    spam_indices = np.where(labels == 1)[0]
    
    # Sample size for each class (use a small batch)
    sample_size = min(20, len(spam_indices), len(real_indices))
    
    # Randomly sample from each class
    np.random.seed(42)
    sampled_real = np.random.choice(real_indices, sample_size, replace=False)
    sampled_spam = np.random.choice(spam_indices, sample_size, replace=False)
    
    # Combine samples with the new message
    update_messages = (
        [messages[i] for i in sampled_real] +  # Real messages
        [messages[i] for i in sampled_spam] +  # Existing spam
        [cleaned_message]  # New spam message
    )
    
    update_labels = np.array(
        [0] * sample_size +  # Labels for real messages
        [1] * sample_size +  # Labels for existing spam
        [1]                  # Label for new message
    )
    
    # Prepare sequences for update
    X_update, _ = prepare_sequences(
        update_messages,
        max_words=10000,
        max_len=max_len,
        tokenizer=tokenizer  # Use existing tokenizer
    )
    
    # Update the model with the balanced batch
    # Use a higher learning rate for quicker adaptation
    model.optimizer.learning_rate.assign(0.001)
    
    history = model.fit(
        X_update, update_labels,
        epochs=training_epochs,
        batch_size=8,
        verbose=0
    )
    
    # Reset learning rate to original value
    model.optimizer.learning_rate.assign(0.0001)
    
    # Return the training history
    return history

def add_spam_message(message):
    """Add a new spam message and update the model"""
    try:
        # Load existing model
        model, tokenizer, max_len = load_model_artifacts()
        
        if model is None:
            raise ValueError("No existing model found. Please train a new model first.")
        
        # Update the dataset
        update_spam_dataset(message)
        
        # Update the model
        history = update_model_with_message(message, model, tokenizer, max_len)
        
        # Save updated model
        save_model_artifacts(model, tokenizer, max_len)
        
        # Test the model with the new message
        prob, is_spam = predict_message(message, model, tokenizer, max_len)
        
        return {
            'success': True,
            'message': 'Model updated successfully',
            'prediction': {
                'probability': float(prob),
                'is_spam': bool(is_spam)
            },
            'training_loss': float(history.history['loss'][-1]),
            'model': model,
            'tokenizer': tokenizer,
            'max_len': max_len
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error updating model: {str(e)}'
        }

def main():
    # Try to load existing model
    print("Loading model...")
    model, tokenizer, saved_max_len = load_model_artifacts()
    
    if model is None:
        print("No existing model found. Training new model...")
        # Load and preprocess data
        print("Loading data...")
        messages, labels = load_data()
        
        # Prepare sequences
        print("Preparing sequences...")
        max_words = 10000
        max_len = 50
        X, tokenizer = prepare_sequences(messages, max_words, max_len)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        # Create and train model
        print("Training model...")
        model = create_model(max_words, max_len)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Generate predictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Spam']))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model artifacts
        print("\nSaving model artifacts...")
        save_model_artifacts(model, tokenizer, max_len)
        saved_max_len = max_len
    
    print("\nSpam Detection System Ready!")
    print("Enter messages to check for spam. Type 'quit' to exit.")
    print("After each prediction, you can mark false positives/negatives to improve the model.")
    
    while True:
        # Get user input
        print("\nEnter a message (or 'quit' to exit):")
        message = input().strip()
        
        if message.lower() in ['quit', 'q', 'exit']:
            break
        
        # Make prediction
        prob, is_spam = predict_message(message, model, tokenizer, saved_max_len)
        
        # Display results
        print(f"\nSpam Probability: {prob:.2%}")
        print(f"Classification: {'SPAM' if is_spam else 'NOT SPAM'}")
        
        # Ask for feedback
        while True:
            print("\nIs this classification correct? (y/n):")
            feedback = input().strip().lower()
            if feedback in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' or 'n'")
        
        # If classification was wrong and it was actually spam, update the model
        if feedback in ['n', 'no'] and not is_spam:
            print("\nUpdating model to recognize this as spam...")
            result = add_spam_message(message)
            if result['success']:
                print("Model updated successfully!")
                print(f"New spam probability: {result['prediction']['probability']:.2%}")
                # Update the model, tokenizer, and max_len with the newly trained versions
                model = result['model']
                tokenizer = result['tokenizer']
                saved_max_len = result['max_len']
            else:
                print(f"Error updating model: {result['message']}")
        elif feedback in ['n', 'no'] and is_spam:
            print("\nMessage will be considered as a false positive.")
            # TODO: Implement handling of false positives
            print("Note: False positive handling will be implemented in a future update.")
    
    print("\nThank you for using the Spam Detection System!")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

def clean_text(text):
    """Minimal text cleaning to preserve emojis and special characters"""
    # Only remove excessive whitespace and standardize to lowercase
    return text.lower().strip()

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

def prepare_sequences(messages, max_words=10000, max_len=50):
    # Create and fit tokenizer
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

def main():
    # Try to load existing model
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
        
    else:
        print("Loaded existing model!")
        # Test the model with a few example predictions
        examples = [
            "Hello, how are you doing today?",
            " DM me now for easy money! 100% guaranteed wins! ",
            "When is the next meeting scheduled?",
            "Send FR for instant profits!  Don't miss out! "
        ]
        
        print("\nTesting model with example messages:")
        for msg in examples:
            prob, is_spam = predict_message(msg, model, tokenizer, saved_max_len)
            print(f"\nMessage: {msg}")
            print(f"Spam Probability: {prob:.2%}")
            print(f"Classification: {'SPAM' if is_spam else 'NOT SPAM'}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

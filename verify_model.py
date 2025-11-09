import joblib
import os

print("=" * 50)
print("CHECKING MODEL FILES")
print("=" * 50)

# Check if files exist
model_exists = os.path.exists('spam_classifier_model.joblib')
vectorizer_exists = os.path.exists('tfidf_vectorizer.joblib')

print(f"\n✓ Model file exists: {model_exists}")
print(f"✓ Vectorizer file exists: {vectorizer_exists}")

if not model_exists or not vectorizer_exists:
    print("\n❌ ERROR: Model files are missing!")
    print("Please run the training code in Colab first and download the files.")
    exit()

# Try to load the files
print("\n" + "=" * 50)
print("LOADING FILES")
print("=" * 50)

try:
    model = joblib.load('spam_classifier_model.joblib')
    print("✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("✓ Vectorizer loaded successfully")
    print(f"  Vectorizer type: {type(vectorizer).__name__}")
except Exception as e:
    print(f"❌ Error loading vectorizer: {e}")
    exit()

# Check if vectorizer is fitted
print("\n" + "=" * 50)
print("CHECKING VECTORIZER")
print("=" * 50)

try:
    # List all attributes
    print("\nVectorizer attributes:")
    attrs = [attr for attr in dir(vectorizer) if not attr.startswith('_')]
    for attr in ['idf_', 'vocabulary_', 'stop_words_']:
        if hasattr(vectorizer, attr):
            value = getattr(vectorizer, attr)
            if attr == 'idf_':
                print(f"  ✓ {attr}: exists (length: {len(value)})")
            elif attr == 'vocabulary_':
                print(f"  ✓ {attr}: exists (size: {len(value)})")
            else:
                print(f"  ✓ {attr}: exists")
        else:
            print(f"  ❌ {attr}: MISSING")
    
    # Check if vectorizer is fitted
    if hasattr(vectorizer, 'idf_'):
        print("\n✓ Vectorizer is fitted (has idf_ attribute)")
        print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"  Number of features: {len(vectorizer.idf_)}")
    else:
        print("\n❌ Vectorizer is NOT fitted (missing idf_ attribute)")
        print("\nDEBUG INFO:")
        print(f"  Type: {type(vectorizer)}")
        print(f"  Module: {type(vectorizer).__module__}")
        
        # Try to check internal state
        if hasattr(vectorizer, '_tfidf'):
            print(f"  Has _tfidf: Yes")
            if hasattr(vectorizer._tfidf, 'idf_'):
                print(f"  _tfidf.idf_ exists: Yes (length: {len(vectorizer._tfidf.idf_)})")
            else:
                print(f"  _tfidf.idf_ exists: No")
        
        print("\nYou need to retrain the model using the fixed training code!")
        exit()
except Exception as e:
    print(f"❌ Error checking vectorizer: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Test prediction
print("\n" + "=" * 50)
print("TESTING PREDICTION")
print("=" * 50)

test_messages = [
    "WINNER!! You have been selected to receive a £900 prize reward!",
    "Hi, how are you doing today? Let's meet for coffee."
]

for i, message in enumerate(test_messages, 1):
    print(f"\nTest {i}: {message[:50]}...")
    try:
        features = vectorizer.transform([message])
        prediction = model.predict(features)
        proba = model.predict_proba(features)
        
        result = "HAM (Legitimate)" if prediction[0] == 1 else "SPAM"
        confidence = proba[0][prediction[0]] * 100
        
        print(f"  Result: {result}")
        print(f"  Confidence: {confidence:.2f}%")
        print("  ✓ Success!")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)
print("\nIf all tests passed, your Streamlit app should work!")
print("If any tests failed, please retrain using the fixed training code.")
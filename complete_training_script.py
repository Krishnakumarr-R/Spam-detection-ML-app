import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import pickle

print("=" * 70)
print("SPAM CLASSIFIER TRAINING - COMPLETE REBUILD")
print("=" * 70)

# Step 1: Load data
print("\n[1/8] Loading data...")
raw_mail_data = pd.read_csv('mail_data.csv')
print(f"✓ Loaded {len(raw_mail_data)} emails")

# Step 2: Clean data
print("\n[2/8] Cleaning data...")
mail_data = raw_mail_data.fillna('')
mail_data = mail_data.drop_duplicates()
print(f"✓ Cleaned data: {len(mail_data)} emails")

# Step 3: Prepare labels
print("\n[3/8] Preparing labels...")
X = mail_data['Message'].values
Y_raw = mail_data['Category'].values

# Convert labels to numeric
label_map = {'spam': 0, 'ham': 1}
Y = np.array([label_map.get(str(label).lower(), 0) for label in Y_raw])

print(f"✓ Total samples: {len(X)}")
print(f"✓ Spam (0): {np.sum(Y == 0)}")
print(f"✓ Ham (1): {np.sum(Y == 1)}")

# Step 4: Split data
print("\n[4/8] Splitting data...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f"✓ Training: {len(X_train)} samples")
print(f"✓ Testing: {len(X_test)} samples")

# Step 5: Create and fit vectorizer - THE CRITICAL PART
print("\n[5/8] Creating TF-IDF Vectorizer...")
print("=" * 70)
print("CRITICAL: FITTING THE VECTORIZER")
print("=" * 70)

# Create fresh vectorizer
vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True,
    max_features=3000
)

# FIT_TRANSFORM - This is what creates the idf_ attribute
print("\nCalling fit_transform on training data...")
X_train_tfidf = vectorizer.fit_transform(X_train)
print(f"✓ fit_transform completed")
print(f"✓ Training features shape: {X_train_tfidf.shape}")

# VERIFY the vectorizer is fitted
print("\n" + "=" * 70)
print("VERIFYING VECTORIZER IS FITTED")
print("=" * 70)

checks_passed = 0
checks_total = 4

# Check 1: vocabulary_
if hasattr(vectorizer, 'vocabulary_') and vectorizer.vocabulary_:
    print(f"✓ Check 1/4: vocabulary_ exists ({len(vectorizer.vocabulary_)} words)")
    checks_passed += 1
else:
    print("❌ Check 1/4: vocabulary_ MISSING")

# Check 2: idf_
if hasattr(vectorizer, 'idf_'):
    print(f"✓ Check 2/4: idf_ exists (direct attribute)")
    checks_passed += 1
else:
    print("❌ Check 2/4: idf_ MISSING from vectorizer")

# Check 3: _tfidf.idf_
if hasattr(vectorizer, '_tfidf') and hasattr(vectorizer._tfidf, 'idf_'):
    print(f"✓ Check 3/4: _tfidf.idf_ exists ({len(vectorizer._tfidf.idf_)} features)")
    checks_passed += 1
else:
    print("❌ Check 3/4: _tfidf.idf_ MISSING")

# Check 4: Can transform
try:
    test_transform = vectorizer.transform(["This is a test message"])
    print(f"✓ Check 4/4: Can transform new text")
    checks_passed += 1
except Exception as e:
    print(f"❌ Check 4/4: Cannot transform - {e}")

print(f"\nPassed {checks_passed}/{checks_total} checks")

if checks_passed < 4:
    print("\n❌ VECTORIZER NOT PROPERLY FITTED - STOPPING")
    raise Exception("Vectorizer fitting failed!")

print("\n✓✓✓ VECTORIZER IS FULLY FITTED ✓✓✓")

# Transform test data
print("\n[6/8] Transforming test data...")
X_test_tfidf = vectorizer.transform(X_test)
print(f"✓ Test features shape: {X_test_tfidf.shape}")

# Step 6: Train model
print("\n[7/8] Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, Y_train)
print("✓ Model trained")

# Evaluate
train_pred = model.predict(X_train_tfidf)
test_pred = model.predict(X_test_tfidf)
train_acc = accuracy_score(Y_train, train_pred)
test_acc = accuracy_score(Y_test, test_pred)

print(f"\n✓ Training Accuracy: {train_acc*100:.2f}%")
print(f"✓ Test Accuracy: {test_acc*100:.2f}%")

# Test predictions
print("\n" + "=" * 70)
print("TESTING SAMPLE PREDICTIONS")
print("=" * 70)

samples = [
    "WINNER! You won $1000! Call now!",
    "Hi, let's meet for dinner tonight",
    "Congratulations! Claim your prize now!",
    "The meeting is at 3pm tomorrow"
]

for i, text in enumerate(samples, 1):
    features = vectorizer.transform([text])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    label = "SPAM" if pred == 0 else "HAM"
    conf = proba[pred] * 100
    print(f"\n{i}. {text[:50]}")
    print(f"   → {label} ({conf:.1f}% confidence)")

# Step 7: Save files
print("\n" + "=" * 70)
print("[8/8] SAVING FILES")
print("=" * 70)

# Save with joblib (using compress for compatibility)
print("\nSaving with joblib...")
joblib.dump(model, 'spam_classifier_model.joblib', compress=3)
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib', compress=3)
print("✓ Saved: spam_classifier_model.joblib")
print("✓ Saved: tfidf_vectorizer.joblib")

# Also save with pickle using protocol 4 for better compatibility
print("\nSaving backup with pickle (protocol 4)...")
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f, protocol=4)
print("✓ Saved: spam_classifier_model.pkl")
print("✓ Saved: tfidf_vectorizer.pkl")

# CRITICAL: Reload and verify
print("\n" + "=" * 70)
print("VERIFYING SAVED FILES")
print("=" * 70)

print("\nLoading joblib files...")
loaded_model = joblib.load('spam_classifier_model.joblib')
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')

verification_passed = True

# Verify model
if loaded_model is not None:
    print("✓ Model loaded successfully")
else:
    print("❌ Model load failed")
    verification_passed = False

# Verify vectorizer attributes
if hasattr(loaded_vectorizer, 'vocabulary_'):
    print(f"✓ Loaded vectorizer has vocabulary_ ({len(loaded_vectorizer.vocabulary_)} words)")
else:
    print("❌ Loaded vectorizer missing vocabulary_")
    verification_passed = False

if hasattr(loaded_vectorizer, '_tfidf') and hasattr(loaded_vectorizer._tfidf, 'idf_'):
    print(f"✓ Loaded vectorizer has _tfidf.idf_ ({len(loaded_vectorizer._tfidf.idf_)} features)")
else:
    print("❌ Loaded vectorizer missing _tfidf.idf_")
    verification_passed = False

# Final test with loaded objects
try:
    test_text = ["Free prize money! Call now!"]
    test_features = loaded_vectorizer.transform(test_text)
    test_prediction = loaded_model.predict(test_features)
    print(f"✓ Prediction test successful: {test_prediction[0]}")
except Exception as e:
    print(f"❌ Prediction test failed: {e}")
    verification_passed = False

if not verification_passed:
    print("\n❌ VERIFICATION FAILED - DO NOT USE THESE FILES")
    raise Exception("Saved files verification failed!")

print("\n✓✓✓ ALL VERIFICATIONS PASSED ✓✓✓")

# Download files
print("\n" + "=" * 70)
print("DOWNLOADING FILES")
print("=" * 70)

try:
    from google.colab import files
    files.download('spam_classifier_model.joblib')
    files.download('tfidf_vectorizer.joblib')
    print("\n✓ Files downloaded!")
except:
    print("Not in Colab - files saved locally")

print("\n" + "=" * 70)
print("✓✓✓ TRAINING COMPLETE AND VERIFIED ✓✓✓")
print("=" * 70)
print("\nYour files are ready to use!")
print("Both .joblib and .pkl versions were created.")
print("\nNext steps:")
print("1. Place the .joblib files in your Streamlit folder")
print("2. Run: python verify_model.py")
print("3. Run: streamlit run app.py")
print("=" * 70)
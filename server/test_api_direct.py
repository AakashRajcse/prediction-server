"""
Direct API testing (without running the server)
This script tests the API functions directly
"""
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from api.utils import initialize_ml, get_prediction, validate_review, validate_rating

def test_direct():
    """Test API functions without server"""
    print("\n" + "="*60)
    print("FAKE REVIEW DETECTION API - DIRECT TEST")
    print("="*60)

    # Initialize ML
    print("\n[1] Initializing ML system...")
    try:
        initialize_ml()
        print("[OK] ML system initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    # Test cases
    test_cases = [
        {
            "review": "This is an excellent product that works perfectly!",
            "rating": 5,
            "expectation": "Likely GENUINE"
        },
        {
            "review": "Amazing excellent fantastic perfect best product ever highly recommend",
            "rating": 5,
            "expectation": "Likely SUSPICIOUS or FAKE"
        },
        {
            "review": "Product works as described, good quality, happy with purchase",
            "rating": 4,
            "expectation": "Likely GENUINE"
        },
        {
            "review": "Terrible waste of money, broke immediately",
            "rating": 1,
            "expectation": "Likely GENUINE"
        }
    ]

    print("\n[2] Testing predictions...")
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}] {test['expectation']}")
        print(f"Review: {test['review'][:60]}...")
        print(f"Rating: {test['rating']}/5")

        # Validate input
        is_valid, error = validate_review(test["review"])
        if not is_valid:
            print(f"[ERROR] Invalid review: {error}")
            continue

        is_valid, error = validate_rating(test["rating"])
        if not is_valid:
            print(f"[ERROR] Invalid rating: {error}")
            continue

        # Get prediction
        try:
            result = get_prediction(test["review"], test["rating"])

            print(f"\nResults:")
            print(f"  Fake Score:    {result['fake_score']:.2%}")
            print(f"  Trust Score:   {result['trust_score']:.2%}")
            print(f"  Verdict:       {result['verdict']}")
            print(f"  Confidence:    {result['confidence']:.1f}%")
            print(f"\nFeatures:")
            print(f"  Words:         {result['word_count']}")
            print(f"  Sentiment:     {result['sentiment']:.3f}")
            print(f"  Exclamations:  {result['exclamation_count']}")
            print(f"  Caps Ratio:    {result['caps_ratio']:.3f}")
            print(f"  Repetition:    {result['repetition']}")

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nTo run the API server, execute:")
    print("  python run_api.py")
    print("\nThen visit: http://localhost:8000/docs")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_direct()

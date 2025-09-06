from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

def build_hybrid_classifier():
    hybrid = VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting="soft"
    )
    return hybrid

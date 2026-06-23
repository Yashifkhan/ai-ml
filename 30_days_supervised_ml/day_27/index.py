import joblib
import pandas as pd

loaded_lr = joblib.load("student_lr.pkl")
loaded_rf = joblib.load("student_rf.pkl")

# test_data = pd.DataFrame([{
#     "student_id": 3,
#     "age": 25,
#     "country": "India",
#     "prior_programming_experience": "Beginner",
#     "weeks_in_course": 10,
#     "hours_spent_learning_per_week": 15.5,
#     "practice_problems_solved": 80,
#     "projects_completed": 2,
#     "tutorial_videos_watched": 40,
#     "uses_kaggle": 1,
#     "participates_in_discussion_forums": 1,
#     "debugging_sessions_per_week": 3,
#     "self_reported_confidence_python": 7
# }])


test_data = pd.DataFrame([
    {
        "student_id": 3,
        "age": 25,
        "country": "India",
        "prior_programming_experience": "Beginner",
        "weeks_in_course": 10,
        "hours_spent_learning_per_week": 15.5,
        "practice_problems_solved": 80,
        "projects_completed": 2,
        "tutorial_videos_watched": 40,
        "uses_kaggle": 1,
        "participates_in_discussion_forums": 1,
        "debugging_sessions_per_week": 3,
        "self_reported_confidence_python": 7
    },
    {
        "student_id": 4,
        "age": 30,
        "country": "USA",
        "prior_programming_experience": "Beginner",
        "weeks_in_course": 8,
        "hours_spent_learning_per_week": 6.0,
        "practice_problems_solved": 20,
        "projects_completed": 0,
        "tutorial_videos_watched": 10,
        "uses_kaggle": 0,
        "participates_in_discussion_forums": 0,
        "debugging_sessions_per_week": 1,
        "self_reported_confidence_python": 3
    }
])

print("LR Prediction:", loaded_lr.predict(test_data))
print("RF Prediction:", loaded_rf.predict(test_data))
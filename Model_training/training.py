import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("augmented_dataset_matched.csv")

X = df.drop(['rating', 'student_id'], axis=1)
y = df['rating']

categorical_features = ['clgbranch', 'clgyear', 'clgsem', 'video_title']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

joblib.dump(pipeline, 'course_recommender.pkl')
print("✅ Model trained and saved as 'course_recommender.pkl'")

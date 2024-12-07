from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pg8000
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load environment variables
load_dotenv()

# Database credentials
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 6543))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# FastAPI app initialization
app = FastAPI()

category_mapping = {
    1: "categorySalad",
    2: "categoryAppetizers",
    3: "categoryBeef",
    4: "categoryBreakfast",
    5: "categoryChicken",
    6: "categoryMixed",
    7: "categoryNoodle",
    8: "categoryPasta",
    9: "categoryPizza",
    10: "categoryPork",
    11: "categorySandwiches",
    12: "categorySeafood",
    13: "categorySidedish",
    14: "categoryVegetable"
}

def fetch_data_user_data():
    connection = None
    try:
        # Establish a connection to the PostgreSQL database
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()

        # Fetch user allergies
        cursor.execute("""
            SELECT egg_free, gluten_free, dairy_free, fish_free, shellfish_free, peanut_free, treenut_free, soy_free, wheat_free
            FROM app_user.user_allergies;
        """)
        df_allergies = pd.DataFrame(cursor.fetchall(), columns=[
            'egg_free', 'gluten_free', 'dairy_free', 'fish_free', 'shellfish_free', 'peanut_free', 
            'treenut_free', 'soy_free', 'wheat_free'
        ])

        # Fetch user health profile
        cursor.execute("""
            SELECT user_id, carbohydrate, fat, protein, fiber, sodium, sugar, calories
            FROM app_user.user_health_profile;
        """)
        df_health = pd.DataFrame(cursor.fetchall(), columns=[
            'user_id', 'carbohydrate', 'fat', 'protein', 'fiber', 'sodium', 'sugar', 'calories'
        ])
        for col in ['carbohydrate', 'fat', 'protein', 'fiber', 'sodium', 'sugar', 'calories']:
            df_health[col] = df_health[col] / 3

        # Fetch user category ratings
        cursor.execute("""
            SELECT "userId", "categoryId", rank
            FROM app_user.user_category_rating;
        """)
        df_category = pd.DataFrame(cursor.fetchall(), columns=['userId', 'categoryId', 'rank'])
        df_category['categoryName'] = df_category['categoryId'].map(category_mapping)
        df_category_pivot = df_category.pivot(index='userId', columns='categoryName', values='rank')
        df_category_pivot.reset_index(inplace=True)
        df_category_pivot.drop(columns=['userId'], inplace=True)

        # Combine the DataFrames
        combined_df = pd.concat([df_health, df_allergies, df_category_pivot], axis=1)

        return combined_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

    finally:
        if connection:
            connection.close()

def fetch_and_transform_food_data():
    connection = None
    try:
        # Establish a connection to the PostgreSQL database
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()

        # SQL query to fetch all data from food.food_details
        sql_query = '''
        SELECT *
        FROM food.food_details;
        '''

        cursor.execute(sql_query)
        data = cursor.fetchall()

        # Fetch column names from the cursor description
        column_names = [desc[0] for desc in cursor.description]

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=column_names)

        # **Rename 'id' to 'foodItemId'**
        if 'id' in df.columns:
            df.rename(columns={'id': 'foodItemId'}, inplace=True)
            print("Renamed 'id' to 'foodItemId' in df_food_details.")
        else:
            print("Warning: 'id' column not found in df_food_details. Available columns:", df.columns.tolist())

        # **Optional: Verify the renaming**
        print("df_food_details columns after renaming:", df.columns.tolist())

        # Return the DataFrame with all data
        return df

    except Exception as e:
        print(f"An error occurred in fetch_and_transform_food_data: {e}")
        return pd.DataFrame()

    finally:
        if connection:
            connection.close()


def fetch_and_transform_swipe_data():
    """
    Fetches swipe data from the PostgreSQL database and transforms it into a dictionary
    that maps each user_id to their liked and disliked food items along with categories.

    Returns:
        dict: A dictionary where each key is a user_id and the value is another dictionary
              containing lists of foodItemId, categoryId, and preference (1 for like, 0 for dislike).
    """
    connection = None
    try:
        # Establish a connection to the PostgreSQL database
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()

        # SQL query to fetch data from app_user.food_swipe_history
        sql_query = '''
        SELECT "userId", "foodItemId", "categoryId", "preference"
        FROM app_user.food_swipe_history;
        '''

        cursor.execute(sql_query)
        data = cursor.fetchall()

        # Fetch column names from the cursor description
        column_names = [desc[0] for desc in cursor.description]

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=column_names)

        # Rename columns for consistency
        df.rename(columns={
            "userId": "user_id",
            "foodItemId": "foodItemId",
            "categoryId": "categoryId"
        }, inplace=True)

        # Map categoryId to category names if category_mapping is provided
        if "category_mapping" in globals() and callable(category_mapping):
            df["categoryId"] = df["categoryId"].map(category_mapping)
        else:
            # If no mapping is provided, you can choose to keep the ID or handle accordingly
            pass

        # Map 'like' to 1 and 'dislike' to 0
        df["preference"] = df["preference"].map({
            "like": 1,
            "dislike": 0
        })

        # Handle any unexpected preference values by setting them to a default (e.g., NaN) and dropping
        df = df.dropna(subset=["preference"])

        # Convert preferences to integer type
        df["preference"] = df["preference"].astype(int)

        # Group by user_id and collect foodItemId, categoryId, and preference
        expected_recommendations = {}
        for user_id, group in df.groupby("user_id"):
            expected_recommendations[user_id] = {
                "foodItemId": group["foodItemId"].tolist(),
                "categoryId": group["categoryId"].tolist(),
                "preference": group["preference"].tolist()
            }

        # Return the recommendations dictionary
        return expected_recommendations

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

    finally:
        if connection:
            connection.close()



# Response model
class RecommendationResponse(BaseModel):
    message: str
    data: list

# Recommendation logic (simplified for API use)
# Define specific weights for nutritional features
nutritional_weights = {
    'calories': 0.5,      # 50% of 80% weight
    'protein': 0.2,       # 20% of 80% weight
    'carbohydrates': 0.1, # 10% of 80% weight
    'fat': 0.1,           # 10% of 80% weight
    'fiber': 0.05,        # 5% of 80% weight
    'sugar': 0.05,        # 5% of 80% weight
    'sodium': 0.0         # 0% weight for sodium (not considered for recommendation)
}

# Define weights
nutrition_weight = 0.8
category_weight_percent = 0.2

# Define specific weights for nutritional features
nutritional_weights = {
    'calories': 0.5,      # 50% of 80% weight
    'protein': 0.2,       # 20% of 80% weight
    'carbohydrates': 0.1, # 10% of 80% weight
    'fat': 0.1,           # 10% of 80% weight
    'fiber': 0.05,        # 5% of 80% weight
    'sugar': 0.05,        # 5% of 80% weight
    'sodium': 0.0         # 0% weight for sodium
}

# Define a mapping from standardized nutrient names to combined_df column names
nutrient_column_mapping = {
    'calories': 'calories',
    'protein': 'protein',
    'carbohydrates': 'carbohydrate',  # Singular form
    'fat': 'fat',
    'fiber': 'fiber',
    'sugar': 'sugar',
    'sodium': 'sodium'
}

def recommend_food_for_user(combined_df, df_food_details, expected_recommendation, user_id, top_n=6):
    """
    Generates top N food recommendations for the latest user based on their preferences and nutritional needs,
    giving 80% importance to nutritional needs and 20% to category preferences.

    Parameters:
    - combined_df (pd.DataFrame): DataFrame containing user information, category preferences, and nutritional needs.
    - df_food_details (pd.DataFrame): DataFrame containing food item details and nutritional information.
    - expected_recommendation (dict): Dictionary mapping user_id to expected foodItemId, categoryId, and preference.
    - top_n (int): Number of top recommendations to generate.

    Returns:
    - pd.DataFrame: DataFrame containing the top recommended dishes with rankings and scores.
    """
    # Use the latest user_id from expected_recommendation
    user_id = max(expected_recommendation.keys())  # Get the latest user_id
    print(f"Generating recommendations for User ID: {user_id}")

    # Extract user data
    user_data = combined_df[combined_df['user_id'] == user_id]
    if user_data.empty:
        print(f"No data found for User ID {user_id}.")
        return

    user_data = user_data.iloc[0]
    menu_data = df_food_details.copy()

    # Extract user expected categories and preferences
    expected_food_ids = expected_recommendation[user_id].get('foodItemId', [])
    expected_categories = expected_recommendation[user_id].get('categoryId', [])
    preferences = expected_recommendation[user_id].get('preference', [])

    # Map foodItemId to preference
    food_preference_map = dict(zip(expected_food_ids, preferences))
    menu_data['preference'] = menu_data['foodItemId'].map(food_preference_map).fillna(0).astype(int)

    # Define target variable as composite score
    # Composite Score = (nutrition_weight * nutrient_score) + (category_weight_percent * preference_score)

    # 1. Calculate Preference Score
    menu_data['preference_score'] = menu_data['preference']  # 1 for like, 0 for dislike

    # 2. Calculate Nutrient Match Score
    # Normalize nutritional features based on user needs
    user_nutritional_needs = {}
    missing_nutrients = []
    for nutrient, column_name in nutrient_column_mapping.items():
        value = user_data.get(column_name, None)
        if value is not None and value > 0:
            user_nutritional_needs[nutrient] = value
        else:
            missing_nutrients.append(nutrient)

    if missing_nutrients:
        print(f"Missing or invalid nutritional needs for User ID {user_id}: {missing_nutrients}")
        return

    # Calculate nutrient match ratios, capped at 1.0
    for nutrient in nutritional_weights.keys():
        if nutrient in user_nutritional_needs:
            target = user_nutritional_needs[nutrient]
            match_feature = f'match_{nutrient}'
            menu_data[match_feature] = menu_data.apply(
                lambda row: min(row[nutrient] / target, 1.0) if row[nutrient] > 0 else 0.0,
                axis=1
            )
        else:
            # If the nutrient is not considered, set match to 0
            match_feature = f'match_{nutrient}'
            menu_data[match_feature] = 0.0

    # Weighted nutrient match score
    menu_data['nutrient_score'] = 0.0
    for nutrient, weight in nutritional_weights.items():
        match_feature = f'match_{nutrient}'
        menu_data['nutrient_score'] += menu_data[match_feature] * weight

    # Define composite score
    menu_data['composite_score'] = (nutrition_weight * menu_data['nutrient_score']) + (category_weight_percent * menu_data['preference_score'])

    # Define target variable
    y = menu_data['composite_score']

    # Feature Engineering: Include category preferences
    category_features = [col for col in menu_data.columns if col.startswith("category") and col != 'categoryId']

    # Normalize category weights based on user preferences
    total_category_preference = user_data[category_features].sum()
    if total_category_preference > 0:
        category_weights_user = user_data[category_features] / total_category_preference * category_weight_percent
    else:
        category_weights_user = pd.Series(0.0, index=category_features)

    print(f"Category weights for User ID {user_id}:")
    print(category_weights_user)

    # Apply category weights to menu_data
    menu_data[category_features] = menu_data[category_features].multiply(category_weights_user, axis=1)

    # List of match features
    match_features = [f'match_{nutrient}' for nutrient in nutritional_weights.keys()]

    # Select only relevant numeric features for imputation and scaling
    numeric_features = category_features + match_features + ['preference_score', 'nutrient_score']

    # Check and list features to impute (only numeric)
    features_to_impute = [col for col in numeric_features if menu_data[col].isnull().any()]

    if features_to_impute:
        print(f"Imputing missing values for features: {features_to_impute}")
        imputer = SimpleImputer(strategy='mean')
        # Ensure only numeric data is imputed
        menu_data[features_to_impute] = imputer.fit_transform(menu_data[features_to_impute])
    else:
        print("No missing values found in numeric features.")

    # Feature Scaling
    scaler = StandardScaler()
    # Select features for scaling: category features and match features
    scaling_features = category_features + match_features
    menu_data[scaling_features] = scaler.fit_transform(menu_data[scaling_features])

    # Prepare training data
    X = menu_data[scaling_features]
    y = menu_data['composite_score']

    # Initialize and train the Regression model
    svr = SVR(kernel='rbf')
    svr.fit(X, y)

    # Predict composite scores
    menu_data['predicted_score'] = svr.predict(X)

    # Sort based on predicted scores
    top_recommended_dishes = menu_data.sort_values(by='predicted_score', ascending=False).head(top_n)

    # Assign rankings
    top_recommended_dishes = top_recommended_dishes.copy()
    top_recommended_dishes['Rank'] = range(1, top_n + 1)

    # Prepare the final DataFrame for recommendations
    top_dishes_with_scores = top_recommended_dishes[['Rank', 'foodItemId', 'predicted_score']].copy()
    top_dishes_with_scores.rename(columns={'foodItemId': 'food_detail_id', 'predicted_score': 'score'}, inplace=True)
    top_dishes_with_scores['user_id'] = user_id

    # Push to the database
    connection = None
    try:
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()

        # Insert recommendations into the database
        for _, row in top_dishes_with_nutrients.iterrows():
            query = '''
            INSERT INTO app_user.user_food_recommendations (user_id, food_detail_id, rank)
            VALUES (%s, %s, %s)
            '''
            cursor.execute(query, (row['user_id'], row['food_detail_id'], row['Rank']))

        connection.commit()
        print("Top recommended dishes have been successfully pushed to the database.")
    except Exception as e:
        print(f"An error occurred while pushing recommendations to the database: {e}")
    finally:
        if connection:
            connection.close()

    print("\nTop Recommended Dishes with Nutritional Information:")
    print(top_dishes_with_nutrients)

    # Return the DataFrame


# API Endpoint
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations():
    """
    Endpoint to get food recommendations for the latest user.
    """
    combined_df = fetch_data_user_data()  
    df_food_details = fetch_and_transform_food_data()
    expected_recommendation = fetch_and_transform_swipe_data() 
    try:
        # Determine the user_id internally
        if not expected_recommendation:
            raise HTTPException(status_code=404, detail="No recommendations available.")

        # For demonstration, select the latest user_id
        user_id = max(expected_recommendation.keys())
        top_n = 6  # Default value; adjust as needed

        print(f"Selected User ID for recommendations: {user_id}")

        recommendations_df = recommend_food_for_user(
            combined_df=combined_df,
            df_food_details=df_food_details,
            expected_recommendation=expected_recommendation,
            user_id=user_id,       # Pass the internally determined user_id
            top_n=top_n
        )

        if recommendations_df.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations available for user {user_id}")

        # Convert the recommendations to a list of dictionaries
        data = recommendations_df.to_dict(orient='records')

        return {"message": "Recommendations fetched successfully", "data": data}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

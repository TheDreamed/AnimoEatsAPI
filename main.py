from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pg8000
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import aiohttp
import joblib  # For model serialization

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

def fetch_data_user_data(user_id):
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
            FROM app_user.user_allergies
            WHERE user_id = %s;
        """, (user_id,))
        df_allergies = pd.DataFrame(cursor.fetchall(), columns=[
            'egg_free', 'gluten_free', 'dairy_free', 'fish_free', 'shellfish_free', 'peanut_free', 
            'treenut_free', 'soy_free', 'wheat_free'
        ])

        # Fetch user health profile
        cursor.execute("""
            SELECT user_id, carbohydrate, fat, protein, fiber, sodium, sugar, calories
            FROM app_user.user_health_profile
            WHERE user_id = %s;
        """, (user_id,))
        df_health = pd.DataFrame(cursor.fetchall(), columns=[
            'user_id', 'carbohydrate', 'fat', 'protein', 'fiber', 'sodium', 'sugar', 'calories'
        ])
        for col in ['carbohydrate', 'fat', 'protein', 'fiber', 'sodium', 'sugar', 'calories']:
            df_health[col] = df_health[col] / 3

        # Fetch user category ratings
        cursor.execute("""
            SELECT "userId", "categoryId", rank
            FROM app_user.user_category_rating
            WHERE "userId" = %s;
        """, (user_id,))
        df_category = pd.DataFrame(cursor.fetchall(), columns=['userId', 'categoryId', 'rank'])
        df_category['categoryName'] = df_category['categoryId'].map(category_mapping)
        df_category_pivot = df_category.pivot(index='userId', columns='categoryName', values='rank')
        df_category_pivot.reset_index(inplace=True)
        df_category_pivot.drop(columns=['userId'], inplace=True)

        # Combine the DataFrames
        combined_df = pd.concat([df_health, df_allergies, df_category_pivot], axis=1)
        print(combined_df)
        return combined_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

    finally:
        if connection:
            connection.close()

def fetch_and_transform_food_data(user_id):
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
        print(df)
        return df

    except Exception as e:
        print(f"An error occurred in fetch_and_transform_food_data: {e}")
        return pd.DataFrame()

    finally:
        if connection:
            connection.close()


def fetch_and_transform_swipe_data(user_id):
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
        FROM app_user.food_swipe_history
        WHERE "userId" = %s;
        '''

        cursor.execute(sql_query, (user_id,))
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
        if "category_mapping" in globals():
            df["categoryName"] = df["categoryId"].map(category_mapping)
        else:
            # If no mapping is provided, you can choose to keep the ID or handle accordingly
            df["categoryName"] = df["categoryId"]

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
        for uid, group in df.groupby("user_id"):
            expected_recommendations[uid] = {
                "foodItemId": group["foodItemId"].tolist(),
                "categoryId": group["categoryId"].tolist(),
                "preference": group["preference"].tolist()
            }
        print(expected_recommendations)
        # Return the recommendations dictionary
        return expected_recommendations

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

    finally:
        if connection:
            connection.close()


class RecommendationRequest(BaseModel):
    user_id: int

# Response model
class RecommendationResponse(BaseModel):
    message: str
    data: list



# Define weights
nutrition_weight = 0.8
category_weight_percent = 0.2

# Define specific weights for nutritional features
nutritional_weights = {
    'calories': 0.2,      # 50% of 80% weight
    'protein': 0.3,       # 20% of 80% weight
    'carbohydrates': 0.5, # 10% of 80% weight
    'fat': 0.2,           # 10% of 80% weight
    'fiber': 0.00,        # 5% of 80% weight
    'sugar': 0.00,        # 5% of 80% weight
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

def recommend_food_for_user(combined_df, df_food_details, expected_recommendation, user_id, top_n=20):
    """
    Generates top N food recommendations for the specified user based on their preferences and nutritional needs,
    using SVC for categorical preferences and SVR for nutritional matching, then combining them.
    Also prints evaluation metrics for both models.

    Parameters:
    - combined_df (pd.DataFrame): DataFrame containing user information, category preferences, and nutritional needs.
    - df_food_details (pd.DataFrame): DataFrame containing food item details and nutritional information.
    - expected_recommendation (dict): Dictionary mapping user_id to expected foodItemId, categoryId, and preference.
    - user_id (int): The ID of the user for whom to generate recommendations.
    - top_n (int): Number of top recommendations to generate.

    Returns:
    - pd.DataFrame: DataFrame containing the top recommended dishes with rankings and scores.
    """
    print(f"Generating recommendations for User ID: {user_id}")

    # Extract user data
    user_data = combined_df[combined_df['user_id'] == user_id]
    if user_data.empty:
        print(f"No data found for User ID {user_id}.")
        return pd.DataFrame()  # Return empty DataFrame

    user_data = user_data.iloc[0]
    menu_data = df_food_details.copy()

    # Extract user expected categories and preferences
    expected_food_ids = expected_recommendation.get(user_id, {}).get('foodItemId', [])
    expected_categories = expected_recommendation.get(user_id, {}).get('categoryId', [])
    preferences = expected_recommendation.get(user_id, {}).get('preference', [])

    # Map foodItemId to preference
    food_preference_map = dict(zip(expected_food_ids, preferences))
    menu_data['preference'] = menu_data['foodItemId'].map(food_preference_map).fillna(0).astype(int)

    # Check if user has enough preferences to train SVC
    if menu_data['preference'].nunique() < 2:
        print("Not enough preference data to train SVC.")
        return pd.DataFrame()

    # 1. Calculate Nutrient Match Score (Manual Calculation)
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
        return pd.DataFrame()  # Return empty DataFrame

    # Calculate nutrient match ratios, capped at 1.0
    for nutrient in nutritional_weights.keys():
        if nutrient in user_nutritional_needs:
            target = user_nutritional_needs[nutrient]
            match_feature = f'match_{nutrient}'
            menu_data[match_feature] = menu_data.apply(
                lambda row: min(row.get(nutrient, 0) / target, 1.0) if row.get(nutrient, 0) > 0 else 0.0,
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

    # 2. Train SVC for Preference Prediction
    # Prepare category features
    category_features = [col for col in menu_data.columns if col.startswith("category") and col != 'categoryId']

    # Handle missing category features
    if not category_features:
        print("No category features found for SVC.")
        return pd.DataFrame()

    X_svc = menu_data[category_features]
    y_svc = menu_data['preference']

    # Split data into training and testing sets for SVC
    X_train_svc, X_test_svc, y_train_svc, y_test_svc = train_test_split(
        X_svc, y_svc, test_size=0.2, random_state=42, stratify=y_svc
    )

    # Impute missing values
    imputer_svc = SimpleImputer(strategy='mean')
    X_train_svc_imputed = imputer_svc.fit_transform(X_train_svc)
    X_test_svc_imputed = imputer_svc.transform(X_test_svc)

    # Feature Scaling
    scaler_svc = StandardScaler()
    X_train_svc_scaled = scaler_svc.fit_transform(X_train_svc_imputed)
    X_test_svc_scaled = scaler_svc.transform(X_test_svc_imputed)

    # Initialize and train the SVC model
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    try:
        svc.fit(X_train_svc_scaled, y_train_svc)
    except Exception as e:
        print(f"An error occurred while training SVC: {e}")
        return pd.DataFrame()

    # Predict on the test set and calculate evaluation metrics for SVC
    try:
        y_pred_svc = svc.predict(X_test_svc_scaled)
        accuracy = accuracy_score(y_test_svc, y_pred_svc)
        precision = precision_score(y_test_svc, y_pred_svc, zero_division=0)
        recall = recall_score(y_test_svc, y_pred_svc, zero_division=0)
        f1 = f1_score(y_test_svc, y_pred_svc, zero_division=0)

        print("\nSVC Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    except Exception as e:
        print(f"An error occurred during SVC evaluation: {e}")
        return pd.DataFrame()

    # Retrain SVC on the entire dataset
    try:
        X_svc_imputed_full = imputer_svc.fit_transform(X_svc)
        X_svc_scaled_full = scaler_svc.fit_transform(X_svc_imputed_full)
        svc.fit(X_svc_scaled_full, y_svc)
    except Exception as e:
        print(f"An error occurred while retraining SVC on full data: {e}")
        return pd.DataFrame()

    # Predict preference probabilities on all menu items
    try:
        X_pred_svc_full = X_svc_scaled_full  # Using all menu items
        preference_probs = svc.predict_proba(X_pred_svc_full)[:, 1]  # Probability of class '1' (like)
        menu_data['preference_score'] = preference_probs
    except NotFittedError as e:
        print(f"SVC model is not fitted: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during SVC prediction: {e}")
        return pd.DataFrame()

    # 3. Train SVR for Nutrient Matching
    # Prepare nutrient features
    nutrient_features = [nutrient for nutrient in nutritional_weights.keys() if nutrient in menu_data.columns]
    if not nutrient_features:
        print("No nutrient features found for SVR.")
        return pd.DataFrame()

    X_svr = menu_data[nutrient_features]
    y_svr = menu_data['nutrient_score']

    # Split data into training and testing sets for SVR
    X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
        X_svr, y_svr, test_size=0.2, random_state=42
    )

    # Impute missing values
    imputer_svr = SimpleImputer(strategy='mean')
    X_train_svr_imputed = imputer_svr.fit_transform(X_train_svr)
    X_test_svr_imputed = imputer_svr.transform(X_test_svr)

    # Feature Scaling
    scaler_svr = StandardScaler()
    X_train_svr_scaled = scaler_svr.fit_transform(X_train_svr_imputed)
    X_test_svr_scaled = scaler_svr.transform(X_test_svr_imputed)

    # Initialize and train the SVR model
    svr = SVR(kernel='rbf')
    try:
        svr.fit(X_train_svr_scaled, y_train_svr)
    except Exception as e:
        print(f"An error occurred while training SVR: {e}")
        return pd.DataFrame()

    # Predict on the test set and calculate evaluation metrics for SVR
    try:
        y_pred_svr = svr.predict(X_test_svr_scaled)
        mse = mean_squared_error(y_test_svr, y_pred_svr)
        r2 = r2_score(y_test_svr, y_pred_svr)

        print("\nSVR Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
    except Exception as e:
        print(f"An error occurred during SVR evaluation: {e}")
        return pd.DataFrame()

    # Retrain SVR on the entire dataset
    try:
        X_svr_imputed_full = imputer_svr.fit_transform(X_svr)
        X_svr_scaled_full = scaler_svr.fit_transform(X_svr_imputed_full)
        svr.fit(X_svr_scaled_full, y_svr)
    except Exception as e:
        print(f"An error occurred while retraining SVR on full data: {e}")
        return pd.DataFrame()

    # Predict nutrient scores on all menu items
    try:
        predicted_nutrient_scores = svr.predict(X_svr_scaled_full)
        menu_data['predicted_nutrient_score'] = predicted_nutrient_scores
    except NotFittedError as e:
        print(f"SVR model is not fitted: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during SVR prediction: {e}")
        return pd.DataFrame()

    # 4. Combine SVC and SVR Predictions into Composite Score
    # Define weights for SVC and SVR
    svc_weight = 0.2  # 20% weight for preference
    svr_weight = 0.8  # 80% weight for nutrient matching

    menu_data['composite_score'] = (
        svc_weight * menu_data['preference_score'] +
        svr_weight * menu_data['predicted_nutrient_score']
    )

    # 5. Generate Top N Recommendations
    top_recommended_dishes = menu_data.sort_values(by='composite_score', ascending=False).head(top_n)

    # Assign rankings
    top_recommended_dishes = top_recommended_dishes.copy()
    top_recommended_dishes['Rank'] = range(1, top_n + 1)

    # Prepare the final DataFrame for recommendations
    top_dishes_with_scores = top_recommended_dishes[['Rank', 'foodItemId', 'composite_score']].copy()
    top_dishes_with_scores.rename(columns={'foodItemId': 'food_detail_id', 'composite_score': 'score'}, inplace=True)
    top_dishes_with_scores['user_id'] = user_id

    # Ensure correct data types
    top_dishes_with_scores['user_id'] = top_dishes_with_scores['user_id'].astype(int)
    top_dishes_with_scores['food_detail_id'] = top_dishes_with_scores['food_detail_id'].astype(int)
    top_dishes_with_scores['Rank'] = top_dishes_with_scores['Rank'].astype(int)
    top_dishes_with_scores['score'] = top_dishes_with_scores['score'].astype(float)

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

        # Insert recommendations into the database with upsert to avoid duplicates
        for _, row in top_dishes_with_scores.iterrows():
            # Explicitly cast to Python int to ensure compatibility
            user_id_int = int(row['user_id'])
            food_detail_id_int = int(row['food_detail_id'])
            rank_int = int(row['Rank'])
            score_float = float(row['score'])

            # Debugging statement to verify types and values
            print(f"Inserting user_id: {user_id_int} (type: {type(user_id_int)}), "
                  f"food_detail_id: {food_detail_id_int} (type: {type(food_detail_id_int)}), "
                  f"rank: {rank_int} (type: {type(rank_int)}), score: {score_float} (type: {type(score_float)})")

            query = '''
            INSERT INTO app_user.user_food_recommendations (user_id, food_detail_id, rank, score)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, food_detail_id)
            DO UPDATE SET rank = EXCLUDED.rank, score = EXCLUDED.score;
            '''
            cursor.execute(query, (user_id_int, food_detail_id_int, rank_int, score_float))

        connection.commit()
        print("Top recommended dishes have been successfully pushed to the database.")
    except Exception as e:
        print(f"An error occurred while pushing recommendations to the database: {e}")
    finally:
        if connection:
            connection.close()

    print("\nTop Recommended Dishes with Scores:")
    print(top_dishes_with_scores)
    return top_dishes_with_scores

# API Endpoint
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Endpoint to get food recommendations for a user.
    """
    try:
        user_id = request.user_id  # Assign user_id from request first
        print(f"Received request for User ID: {user_id}")

        # Construct the external URL with the provided user_id
        external_url = f"https://app-ivqcpctaoq-uc.a.run.app/dev/food/{user_id}/available"

        # Fetch data from the external URL
        async with aiohttp.ClientSession() as session:
            async with session.get(external_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to fetch external data."
                    )
                external_data = await response.json()
                # TODO: Process external_data as needed
                # For example, you might integrate external_data into your dataframes

        # Fetch and transform data
        combined_df = fetch_data_user_data(user_id)
        df_food_details = fetch_and_transform_food_data(user_id)
        expected_recommendation = fetch_and_transform_swipe_data(user_id)

        # **Fix starts here**
        if not expected_recommendation:
            raise HTTPException(status_code=404, detail="No recommendations available.")
        # **Fix ends here**

        # Ensure user_id is an integer (already ensured by Pydantic, but converting just in case)
        user_id = int(user_id)
        top_n = 20  # Default value; adjust as needed

        print(f"Generating top {top_n} recommendations for User ID: {user_id}")

        # Generate recommendations
        recommendations_df = recommend_food_for_user(
            combined_df=combined_df,
            df_food_details=df_food_details,
            expected_recommendation=expected_recommendation,
            user_id=user_id,
            top_n=top_n
        )

        if recommendations_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations available for user {user_id}"
            )

        # Convert the recommendations to a list of dictionaries
        data = recommendations_df.to_dict(orient='records')

        return RecommendationResponse(
            message="Recommendations fetched successfully",
            data=data
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise http_exc
    except Exception as e:
        # Log the exception details as needed
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

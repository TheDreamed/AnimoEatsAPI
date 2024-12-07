from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pg8000
import os
from dotenv import load_dotenv
from sklearn.svm import SVC
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

        # Rename columns
        df.rename(columns={"userId": "user_id", "categoryId": "categoryId", "foodItemId": "foodItemId"}, inplace=True)

        # Map categoryId to category names (if category_mapping is provided)
        df["categoryId"] = df["categoryId"].map(category_mapping)

        # Filter rows where preference is 'like' and convert it to 1
        df = df[df["preference"] == "like"]
        df["preference"] = 1

        # Group by user_id and collect foodItemId and categoryId
        expected_recommendation = {}
        for user_id, group in df.groupby("user_id"):
            expected_recommendation[user_id] = {
                "foodItemId": group["foodItemId"].tolist(),
                "categoryId": group["categoryId"].tolist(),
            }

        # Return the recommendations dictionary
        return expected_recommendation

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

def recommend_food_for_user(combined_df, df_food_details, expected_recommendation, top_n=6):
    """
    Generates top N food recommendations for the latest user based on their preferences and pushes the recommendations
    to the app_user.user_food_recommendations table.

    Parameters:
    - combined_df (pd.DataFrame): DataFrame containing user information and category preferences.
    - df_food_details (pd.DataFrame): DataFrame containing food item details.
    - expected_recommendation (dict): Dictionary mapping user_id to expected foodItemId and categoryId.
    - top_n (int): Number of top recommendations to generate.

    Returns:
    - pd.DataFrame: DataFrame containing the top recommended dishes with rankings.
    """
    # Use the latest user_id from expected_recommendation
    user_id = max(expected_recommendation.keys())  # Get the latest user_id
    print(f"Generating recommendations for the latest User ID: {user_id}")

    # Extract user data
    user_data = combined_df[combined_df['user_id'] == user_id]
    if user_data.empty:
        print(f"No data found for User ID {user_id}.")
        return

    user_data = user_data.iloc[0]
    menu_data = df_food_details.copy()

    expected_food_ids = expected_recommendation[user_id]['foodItemId']
    expected_categories = expected_recommendation[user_id]['categoryId']

    # Apply category weights
    category_weight = 0.2  # 20% weight for categories
    category_features = [col for col in menu_data.columns if col.startswith("category") and col != 'categoryId']

    for category in category_features:
        if category in user_data:
            menu_data[category] = menu_data[category] * user_data[category] * category_weight

    # Apply nutritional weights
    nutrition_weight = 0.8  # 80% weight for nutritional features
    nutritional_features = ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugar', 'sodium']
    nutritional_features_present = [feature for feature in nutritional_features if feature in menu_data.columns]

    for feature in nutritional_features_present:
        weight = nutritional_weights.get(feature, 0.0)  # Default to 0.0 if weight not defined
        menu_data[feature] = menu_data[feature] * weight * nutrition_weight

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features_to_impute = category_features + nutritional_features_present
    menu_data[features_to_impute] = imputer.fit_transform(menu_data[features_to_impute])

    # Ensure 'categoryId' is integer
    if 'categoryId' in menu_data.columns:
        menu_data['categoryId'] = menu_data['categoryId'].dropna().astype(int)
    else:
        print("Error: 'categoryId' column is missing in df_food_details.")
        return

    # Scale features
    scaler = StandardScaler()
    menu_data[features_to_impute] = scaler.fit_transform(menu_data[features_to_impute])

    # Prepare training data for SVM
    menu_data['target_category'] = menu_data['categoryId']
    X = menu_data[features_to_impute]
    y = menu_data['target_category']

    # Initialize and train the SVM classifier
    svm = SVC(probability=True, random_state=42)
    svm.fit(X, y)

    # Predict probabilities for each category
    probabilities = svm.predict_proba(X)
    predictions_df = df_food_details.loc[X.index].reset_index(drop=True)

    # Assign predicted categories and their probabilities
    predictions_df['predicted_category'] = svm.predict(X)
    category_labels = svm.classes_
    category_probabilities = pd.DataFrame(probabilities, columns=[f"prob_{cls}" for cls in category_labels])
    predictions_df = pd.concat([predictions_df, category_probabilities], axis=1)

    # Determine the maximum probability for ranking
    predictions_df['max_probability'] = probabilities.max(axis=1)
    top_recommended_dishes = predictions_df.sort_values(by='max_probability', ascending=False).head(top_n)

    # Assign rankings
    top_recommended_dishes['Rank'] = range(1, top_n + 1)

    # Prepare the final DataFrame for recommendations
    top_dishes_with_nutrients = top_recommended_dishes[['Rank', 'foodItemId']].copy()
    top_dishes_with_nutrients.rename(columns={'foodItemId': 'food_detail_id'}, inplace=True)
    top_dishes_with_nutrients['user_id'] = user_id

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

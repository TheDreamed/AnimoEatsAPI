from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import pg8000
import os
from dotenv import load_dotenv
from sklearn.svm import SVR  # Using SVR for regression tasks
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import httpx
import logging
import ssl  # Only if SSL is required

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the request model
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="The ID of the user for whom to generate recommendations.")

# Define the response model
class RecommendationResponse(BaseModel):
    message: str
    data: list

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

# Category mapping
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
            # ssl_context=ssl.create_default_context()  # Uncomment if SSL is required
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

        logger.info("Successfully fetched and combined user data.")
        return combined_df

    except Exception as e:
        logger.error(f"An error occurred in fetch_data_user_data: {e}")
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
            # ssl_context=ssl.create_default_context()  # Uncomment if SSL is required
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
            logger.info("Renamed 'id' to 'foodItemId' in df_food_details.")
        else:
            logger.warning(f"'id' column not found in df_food_details. Available columns: {df.columns.tolist()}")

        # **Ensure 'foodItemId' is integer**
        if 'foodItemId' in df.columns:
            df['foodItemId'] = df['foodItemId'].astype(int)
        else:
            logger.warning("'foodItemId' column is missing after renaming.")

        # **Optional: Verify the renaming**
        logger.info(f"df_food_details columns after renaming: {df.columns.tolist()}")

        # Return the DataFrame with all data
        return df

    except Exception as e:
        logger.error(f"An error occurred in fetch_and_transform_food_data: {e}")
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
            # ssl_context=ssl.create_default_context()  # Uncomment if SSL is required
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

        # Map categoryId to category names
        df["categoryId"] = df["categoryId"].map(category_mapping)

        # Map 'like' to 1 and 'dislike' to 0
        df["preference"] = df["preference"].map({
            "like": 1,
            "dislike": 0
        })

        # Drop rows with NaN preferences
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

        logger.info("Successfully fetched and transformed swipe data.")
        return expected_recommendations

    except Exception as e:
        logger.error(f"An error occurred in fetch_and_transform_swipe_data: {e}")
        return {}

    finally:
        if connection:
            connection.close()

# Fetch data once at startup
combined_df = fetch_data_user_data()  
df_food_details = fetch_and_transform_food_data()
expected_recommendations = fetch_and_transform_swipe_data() 

def recommend_food_for_user(combined_df, df_food_details, expected_recommendation, user_id, top_n=6):
    """
    Generates top N food recommendations for a specified user based on their preferences and nutritional needs,
    giving 80% importance to nutritional needs and 20% to category preferences.

    Parameters:
    - combined_df (pd.DataFrame): DataFrame containing user information and category preferences.
    - df_food_details (pd.DataFrame): DataFrame containing food item details.
    - expected_recommendation (dict): Dictionary mapping user_id to expected foodItemId and categoryId.
    - user_id (int): The ID of the user to generate recommendations for.
    - top_n (int): Number of top recommendations to generate.

    Returns:
    - pd.DataFrame: DataFrame containing the top recommended dishes with rankings and scores.
    """
    logger.info("Combined DataFrame:")
    logger.info(combined_df)
    logger.info("Food Details DataFrame:")
    logger.info(df_food_details)
    logger.info(f"Expected Recommendations for User {user_id}: {expected_recommendation.get(user_id, {})}")

    logger.info(f"Generating recommendations for User ID: {user_id}")

    if user_id not in expected_recommendation:
        logger.warning(f"No expected recommendations found for User ID {user_id}.")
        return pd.DataFrame()

    user_data = combined_df[combined_df['user_id'] == user_id]
    if user_data.empty:
        logger.warning(f"No data found for User ID {user_id}.")
        return pd.DataFrame()

    user_data = user_data.iloc[0]
    menu_data = df_food_details.copy()

    expected_food_ids = expected_recommendation[user_id]['foodItemId']
    expected_categories = expected_recommendation[user_id]['categoryId']

    category_weight = 0.2
    category_features = [col for col in menu_data.columns if col.startswith("category") and col != 'categoryId']

    for category in category_features:
        if category in user_data:
            menu_data[category] = menu_data[category] * user_data[category] * category_weight

    nutritional_features = ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugar', 'sodium']
    nutrition_weight = 0.8
    nutritional_features_present = [feature for feature in nutritional_features if feature in menu_data.columns]

    for feature in nutritional_features_present:
        menu_data[feature] = menu_data[feature] * nutrition_weight

    imputer = SimpleImputer(strategy='mean')
    menu_data[category_features + nutritional_features_present] = imputer.fit_transform(menu_data[category_features + nutritional_features_present])

    if 'categoryId' in menu_data.columns:
        menu_data['categoryId'] = menu_data['categoryId'].dropna().astype(int)
    else:
        logger.warning("categoryId column is missing in menu_data.")
        return pd.DataFrame()

    scaler = StandardScaler()
    scaling_features = category_features + nutritional_features_present
    try:
        menu_data[scaling_features] = scaler.fit_transform(menu_data[scaling_features])
    except ValueError as ve:
        logger.error(f"StandardScaler encountered an error: {ve}")
        return pd.DataFrame()

    # Prepare training data
    X = menu_data[scaling_features]
    y = menu_data['categoryId']  # Using categoryId as target

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

    # Ensure correct data types
    top_dishes_with_scores['user_id'] = top_dishes_with_scores['user_id'].astype(int)
    top_dishes_with_scores['food_detail_id'] = top_dishes_with_scores['food_detail_id'].astype(int)
    top_dishes_with_scores['Rank'] = top_dishes_with_scores['Rank'].astype(int)

    # Push to the database
    connection = None
    try:
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
            # ssl_context=ssl.create_default_context()  # Uncomment if SSL is required
        )
        cursor = connection.cursor()

        # Insert recommendations into the database
        for _, row in top_dishes_with_scores.iterrows():
            query = '''
            INSERT INTO app_user.user_food_recommendations (user_id, food_detail_id, rank)
            VALUES (%s, %s, %s)
            '''
            cursor.execute(query, (row['user_id'], row['food_detail_id'], row['Rank']))

        connection.commit()
        logger.info("Top recommended dishes have been successfully pushed to the database.")
    except Exception as e:
        logger.error(f"An error occurred while pushing recommendations to the database: {e}")
    finally:
        if connection:
            connection.close()

    logger.info("\nTop Recommended Dishes with Scores:")
    logger.info(top_dishes_with_scores)

    # Return the DataFrame
    return top_dishes_with_scores

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Endpoint to get food recommendations for a specified user.
    """
    # Extract user_id from the request
    user_id = request.user_id
    logger.info(f"Received recommendation request for user_id: {user_id}")

    # Construct the external URL with the provided user_id
    external_url = f"https://app-ivqcpctaoq-uc.a.run.app/dev/food/{user_id}/available"

    # Make an asynchronous GET request to the external URL
    async with httpx.AsyncClient() as client:
        try:
            external_response = await client.get(external_url)
            external_response.raise_for_status()
            available_foods = external_response.json()

            logger.info(f"Available foods for user {user_id}: {available_foods}")

            # Ensure available_foods is a list of integers
            if not isinstance(available_foods, list):
                logger.error("External API did not return a list.")
                raise HTTPException(status_code=500, detail="Invalid data format received from external service.")

            try:
                available_foods = [int(food_id) for food_id in available_foods]
            except ValueError:
                logger.error("Non-integer foodItemId values received.")
                raise HTTPException(status_code=500, detail="Invalid foodItemId values received from external service.")

            df_food_details = fetch_and_transform_food_data()
            if available_foods:
                df_food_details = df_food_details[df_food_details['foodItemId'].isin(available_foods)]
                logger.info(f"Filtered food details count: {len(df_food_details)}")
            else:
                logger.warning("No available foods found for the user.")
                raise HTTPException(status_code=404, detail="No available foods found for the user.")

            if df_food_details.empty:
                logger.warning("Filtered food details are empty after applying available_foods.")
                raise HTTPException(status_code=404, detail="No matching food items found for recommendations.")

        except httpx.HTTPStatusError as http_exc:
            logger.error(f"HTTP error while fetching available foods: {http_exc}")
            raise HTTPException(status_code=http_exc.response.status_code, detail=f"Error fetching available foods: {http_exc.response.text}")
        except Exception as e:
            logger.error(f"An error occurred while fetching available foods: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching available foods: {str(e)}")

    # Proceed with existing recommendation logic
    combined_df = fetch_data_user_data()
    expected_recommendation = fetch_and_transform_swipe_data()

    # Verify user existence
    if user_id not in expected_recommendation:
        logger.warning(f"No recommendation data available for user {user_id}")
        raise HTTPException(status_code=404, detail=f"No recommendation data available for user {user_id}")

    if user_id not in combined_df['user_id'].values:
        logger.warning(f"No user health data found for user {user_id}")
        raise HTTPException(status_code=404, detail=f"No user health data found for user {user_id}")

    try:
        top_n = 6  # Default value; adjust as needed

        logger.info(f"Generating top {top_n} recommendations for user_id: {user_id}")

        recommendations_df = recommend_food_for_user(
            combined_df=combined_df,
            df_food_details=df_food_details,
            expected_recommendation=expected_recommendation,
            user_id=user_id,       # Pass the externally provided user_id
            top_n=top_n
        )

        if recommendations_df.empty:
            logger.warning(f"No recommendations available for user {user_id}")
            raise HTTPException(status_code=404, detail=f"No recommendations available for user {user_id}")

        # Convert the recommendations to a list of dictionaries
        data = recommendations_df.to_dict(orient='records')

        logger.info(f"Successfully generated recommendations for user_id {user_id}: {data}")

        return {"message": "Recommendations fetched successfully", "data": data}

    except HTTPException as http_exc:
        logger.error(f"HTTPException: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"An error occurred during recommendation generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

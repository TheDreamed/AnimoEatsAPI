from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pg8000
import os
import ssl
import logging
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_ssl_context():
    # Proper SSL Configuration
    ssl_ctx = ssl.create_default_context()
    # If using a self-signed certificate, load it here
    # ssl_ctx.load_verify_locations(cafile='path/to/ca_certificate.pem')
    return ssl_ctx

def fetch_data_user_data():
    connection = None
    try:
        # Establish a connection to the PostgreSQL database with SSL
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            ssl_context=get_ssl_context()
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
        # Establish a connection to the PostgreSQL database with SSL
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            ssl_context=get_ssl_context()
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

        # Rename 'id' to 'foodItemId'
        if 'id' in df.columns:
            df.rename(columns={'id': 'foodItemId'}, inplace=True)
            logger.info("Renamed 'id' to 'foodItemId' in df_food_details.")
        else:
            logger.warning(f"'id' column not found in df_food_details. Available columns: {df.columns.tolist()}")

        # Verify the renaming
        logger.info(f"df_food_details columns after renaming: {df.columns.tolist()}")

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
        # Establish a connection to the PostgreSQL database with SSL
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            ssl_context=get_ssl_context()
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

        # Filter rows where preference is 'like' and convert it to 1
        df = df[df["preference"] == "like"]
        df["preference"] = 1

        # Group by user_id and collect foodItemId and categoryId
        expected_recommendations = {}
        for user_id, group in df.groupby("user_id"):
            expected_recommendations[user_id] = {
                "foodItemId": group["foodItemId"].tolist(),
                "categoryId": group["categoryId"].tolist(),
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

# Response model
class RecommendationResponse(BaseModel):
    message: str
    data: list

# Recommendation logic (simplified for API use)
def recommend_food_for_user(combined_df, df_food_details, expected_recommendation, user_id, top_n=6):
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
        logger.error("categoryId column is missing in menu_data.")
        return pd.DataFrame()

    scaler = StandardScaler()
    menu_data[category_features + nutritional_features_present] = scaler.fit_transform(menu_data[category_features + nutritional_features_present])

    menu_data['target_category'] = menu_data['categoryId']
    X = menu_data[category_features + nutritional_features_present]
    y = menu_data['target_category']

    svm = SVC(probability=True, random_state=42)
    svm.fit(X, y)

    probabilities = svm.predict_proba(X)
    predictions_df = df_food_details.loc[X.index].reset_index(drop=True)

    predictions_df['predicted_category'] = svm.predict(X)
    category_labels = svm.classes_
    category_probabilities = pd.DataFrame(probabilities, columns=[f"prob_{cls}" for cls in category_labels])
    predictions_df = pd.concat([predictions_df, category_probabilities], axis=1)

    predictions_df['max_probability'] = probabilities.max(axis=1)
    top_recommended_dishes = predictions_df.sort_values(by='max_probability', ascending=False).head(top_n)

    top_recommended_dishes['Rank'] = range(1, top_n + 1)
    top_dishes_with_nutrients = top_recommended_dishes[['Rank', 'foodItemId']]
    top_dishes_with_nutrients.rename(columns={'foodItemId': 'food_detail_id'}, inplace=True)  # Rename column
    top_dishes_with_nutrients['user_id'] = user_id

    # Push to the database
    connection = None
    try:
        connection = pg8000.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            ssl_context=get_ssl_context()
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
        logger.info("Top recommended dishes have been successfully pushed to the database.")
    except Exception as e:
        logger.error(f"An error occurred while pushing recommendations to the database: {e}")
    finally:
        if connection:
            connection.close()

    logger.info("\nTop Recommended Dishes with Nutritional Information:")
    logger.info(top_dishes_with_nutrients)

    # Return the DataFrame
    return top_dishes_with_nutrients

# API Endpoint
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations():
    """
    Endpoint to get food recommendations for the latest user.
    """
    try:
        # Determine the user_id internally
        if not expected_recommendations:
            logger.warning("No recommendations available.")
            raise HTTPException(status_code=404, detail="No recommendations available.")

        # For demonstration, select the latest user_id
        user_id = max(expected_recommendations.keys())
        top_n = 6  # Default value; adjust as needed

        logger.info(f"Selected User ID for recommendations: {user_id}")

        recommendations_df = recommend_food_for_user(
            combined_df=combined_df,
            df_food_details=df_food_details,
            expected_recommendation=expected_recommendations,
            user_id=user_id,       # Pass the internally determined user_id
            top_n=top_n
        )

        if recommendations_df.empty:
            logger.warning(f"No recommendations available for user {user_id}")
            raise HTTPException(status_code=404, detail=f"No recommendations available for user {user_id}")

        # Convert the recommendations to a list of dictionaries
        data = recommendations_df.to_dict(orient='records')

        return {"message": "Recommendations fetched successfully", "data": data}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Root Endpoint to handle HEAD and GET requests to '/'
@app.get("/")
async def root():
    return {"message": "API is up and running."}

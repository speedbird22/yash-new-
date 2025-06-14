import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from PIL import Image
import io
import pandas as pd
import random
import time

# Streamlit config
st.set_page_config(page_title="Next-Gen Dish Recommender + Gamification", layout="wide")
st.title("🍽️ Visual Menu Challenge & Recommendation Platform")

# Credentials Initialization (use your secrets.toml)
try:
    vision_credentials_dict = dict(st.secrets["GOOGLE_CLOUD_VISION_CREDENTIALS"])
    vision_credentials = service_account.Credentials.from_service_account_info(vision_credentials_dict)
    vision_client = vision.ImageAnnotatorClient(credentials=vision_credentials)

    firebase_credentials_dict = dict(st.secrets["FIREBASE_CREDENTIALS"])
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(firebase_credentials_dict))
    db = firestore.client()

    gemini_api_key = st.secrets["GEMINI"]["api_key"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# Utility Functions
@st.cache_data(ttl=300)
def fetch_menu():
    return [doc.to_dict() | {"id": doc.id} for doc in db.collection("menu").stream()]

@st.cache_data(ttl=60)
def fetch_challenge_entries():
    return [doc.to_dict() | {"id": doc.id} for doc in db.collection("visual_challenges").stream()]

def calculate_score(entry):
    base_score = entry.get("views", 0) + entry.get("likes", 0) * 2 + entry.get("orders", 0) * 3
    if entry.get("trendy"): base_score += 5
    if entry.get("diet_match"): base_score += 3
    return base_score

def calculate_rating(entry, user_dietary):
    """Calculate a 1-5 star rating based on demand, popularity, and trends."""
    if not entry:
        return 0, "Not Rated"
    
    # Calculate raw score
    score = entry.get("views", 0) * 1 + entry.get("likes", 0) * 2 + entry.get("orders", 0) * 3
    if entry.get("trendy"): score += 5
    if user_dietary and any(tag in entry.get("ingredients", []) for tag in user_dietary):
        score += 3

    # Normalize to 1-5 stars (assuming max score of 50 for simplicity)
    max_score = 50  # Adjust based on observed max values in your data
    normalized_score = min(max(int((score / max_score) * 5), 1), 5)
    star_rating = "★" * normalized_score + "☆" * (5 - normalized_score)
    return normalized_score, star_rating

# Sidebar Preferences
st.sidebar.header("Customer Preferences")
dietary = st.sidebar.multiselect("Diet", ["Vegan", "Vegetarian", "Keto", "Gluten-Free", "Paleo"], default=[])
allergies = st.sidebar.multiselect("Allergies", ["Nut-Free", "Shellfish-Free", "Soy-Free", "Dairy-Free"], default=[])

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 AI Dish Detection", "🎯 Personalized Menu", "⚙️ Custom Filters", "🏅 Visual Menu Challenge", "📊 Leaderboard"])

# TAB 1: AI Dish Detection
with tab1:
    st.header("Visual Dish Detection (AI + Vision API)")
    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded", use_column_width=True)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=image.format)
        content = img_bytes.getvalue()

        response = vision_client.label_detection(image=vision.Image(content=content))
        labels = [label.description for label in response.label_annotations][:5]
        dish_guess = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            f"Predict the most likely dish from these labels: {labels}"
        ).text.strip()

        # Get rating for predicted dish
        challenge_entries = fetch_challenge_entries()
        dish_entry = next((entry for entry in challenge_entries if entry['dish'].lower() == dish_guess.lower()), None)
        rating, star_rating = calculate_rating(dish_entry, dietary)
        
        st.success(f"Predicted Dish: {dish_guess}\n**Rating**: {star_rating} ({rating}/5)")

# TAB 2: Personalized Menu Recommendations
with tab2:
    st.header("Personalized AI Menu")
    menu = fetch_menu()
    menu_text = "\n".join([
        f"- {item['name']}: {item.get('description', '')} ({', '.join(item.get('dietary_tags', []))})"
        for item in menu
    ])
    user_profile = f"Diet: {', '.join(dietary) if dietary else 'None'}, Allergies: {', '.join(allergies) if allergies else 'None'}"
    prompt = f"""
    Given user profile ({user_profile}) recommend 5 dishes from this menu:
    {menu_text}
    For each dish, provide a brief explanation of why it suits the user's profile.
    Format the response as a list with dish name, description, and reason for recommendation.
    """
    ai_result = gemini_model.generate_content(prompt).text.strip()

    # Add ratings to recommended dishes
    challenge_entries = fetch_challenge_entries()
    formatted_result = ""
    for line in ai_result.split("\n"):
        if line.startswith("- **") or line.startswith("**"):  # Assuming Gemini returns dish names in bold
            dish_name = line.strip("- *").split(":")[0].strip()
            dish_entry = next((entry for entry in challenge_entries if entry['dish'].lower() == dish_name.lower()), None)
            rating, star_rating = calculate_rating(dish_entry, dietary)
            formatted_result += f"{line} **Rating**: {star_rating} ({rating}/5)\n"
        else:
            formatted_result += f"{line}\n"
    st.markdown(formatted_result)

# TAB 3: Custom Filtering Options
with tab3:
    st.header("Custom Menu Filters")
    portion = st.selectbox("Portion Size", ["Regular", "Small", "Large"])
    ingredient_swap = st.text_input("Ingredient Swap")

    filtered_menu = []
    challenge_entries = fetch_challenge_entries()
    for item in menu:
        tags = item.get("dietary_tags", [])
        ingredients = item.get("ingredients", [])
        if (not dietary or any(d in tags for d in dietary)) and \
           (not allergies or all(a not in ingredients for a in allergies)):
            item_copy = item.copy()
            item_copy["portion_size"] = portion
            item_copy["ingredient_swap"] = ingredient_swap
            # Add rating if dish is in challenge_entries
            dish_entry = next((entry for entry in challenge_entries if entry['dish'].lower() == item['name'].lower()), None)
            rating, star_rating = calculate_rating(dish_entry, dietary)
            item_copy["rating"] = star_rating
            item_copy["rating_score"] = rating
            filtered_menu.append(item_copy)
    st.write(pd.DataFrame(filtered_menu))

# TAB 4: Staff Gamification Upload
with tab4:
    st.header("Visual Menu Challenge Submission")

    with st.form("challenge_form"):
        staff_name = st.text_input("Staff Name")
        dish_name = st.text_input("Dish Name")
        ingredients = st.text_area("Ingredients (comma separated)")
        plating_style = st.text_input("Plating Style")
        challenge_image = st.file_uploader("Dish Photo", type=["jpg", "png"])
        trendy = st.checkbox("Matches current food trends")
        diet_match = st.checkbox("Matches dietary preferences")

        submitted = st.form_submit_button("Submit Dish")

        if submitted and challenge_image:
            img_bytes = challenge_image.read()
            img_blob = db.collection("visual_challenges").document()
            img_blob.set({
                "staff": staff_name,
                "dish": dish_name,
                "ingredients": [i.strip() for i in ingredients.split(",")],
                "style": plating_style,
                "trendy": trendy,
                "diet_match": diet_match,
                "timestamp": time.time(),
                "views": 0,
                "likes": 0,
                "orders": 0
            })
            st.success("Dish submitted successfully!")

# TAB 5: Leaderboard & Customer Feedback
with tab5:
    st.header("Leaderboard & Voting")

    entries = fetch_challenge_entries()

    for entry in entries:
        with st.container():
            rating, star_rating = calculate_rating(entry, dietary)
            st.subheader(f"{entry['dish']} by {entry['staff']} ({star_rating})")
            st.write(f"Style: {entry['style']}")
            st.write(f"Ingredients: {', '.join(entry['ingredients'])}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"❤️ Like ({entry['likes']})", key=f"like_{entry['id']}"):
                    db.collection("visual_challenges").document(entry['id']).update({"likes": entry['likes'] + 1})
                    st.experimental_rerun()
            with col2:
                if st.button(f"👀 View ({entry['views']})", key=f"view_{entry['id']}"):
                    db.collection("visual_challenges").document(entry['id']).update({"views": entry['views'] + 1})
                    st.experimental_rerun()
            with col3:
                if st.button(f"🛒 Order ({entry['orders']})", key=f"order_{entry['id']}"):
                    db.collection("visual_challenges").document(entry['id']).update({"orders": entry['orders'] + 1})
                    st.experimental_rerun()

    # Show leaderboard
    st.subheader("🏆 Live Leaderboard")
    leaderboard = sorted(entries, key=lambda e: calculate_rating(e, dietary)[0], reverse=True)
    for i, entry in enumerate(leaderboard[:5]):
        rating, star_rating = calculate_rating(entry, dietary)
        st.write(f"**#{i+1} - {entry['dish']} by {entry['staff']} → {star_rating} ({rating}/5, Score: {calculate_score(entry)} pts)**")

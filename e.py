import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from PIL import Image, ImageEnhance
import io
import pandas as pd
import random
import time
import uuid

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

# Sidebar Preferences
st.sidebar.header("Customer Preferences")
dietary = st.sidebar.multiselect("Diet", ["Vegan", "Vegetarian", "Keto", "Gluten-Free", "Paleo"], default=[])
allergies = st.sidebar.multiselect("Allergies", ["Nut-Free", "Shellfish-Free", "Soy-Free", "Dairy-Free"], default=[])

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 AI Dish Detection", "🎯 Personalized Menu", "⚙️ Custom Filters", "🏅 Visual Menu Challenge", "📊 Leaderboard"])

# TAB 1: AI Dish Detection (Updated)
with tab1:
    st.header("Visual Dish Detection (AI + Vision API)")
    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        content = img_bytes.getvalue()

        # Vision API: Label detection and object localization
        response = vision_client.label_detection(image=vision.Image(content=content))
        labels = [label.description for label in response.label_annotations if label.score > 0.7]
        obj_response = vision_client.object_localization(image=vision.Image(content=content))
        objects = [obj.name for obj in obj_response.localized_object_annotations]

        # Combine labels and objects
        combined_labels = list(set(labels + objects))
        st.write(f"Detected Labels and Objects: {combined_labels}")

        # Check if food-related
        food_related = any(label.lower() in ["food", "dish", "meal"] or "food" in label.lower() for label in combined_labels)
        if not food_related:
            st.warning("The image doesn't appear to contain food. Please upload a food-related image.")
            st.stop()

        # Cross-reference with menu
        menu = fetch_menu()
        menu_text = "\n".join([f"{item['name']}: {item.get('description', '')} ({', '.join(item.get('dietary_tags', []))})" for item in menu])
        user_profile = f"Diet: {', '.join(dietary) if dietary else 'None'}, Allergies: {', '.join(allergies) if allergies else 'None'}"
        matching_dishes = []
        for item in menu:
            ingredients = item.get('ingredients', [])
            tags = item.get('dietary_tags', [])
            if any(label.lower() in ' '.join(ingredients + tags).lower() for label in combined_labels):
                matching_dishes.append(item['name'])

        # Gemini prompt with context
        prompt = f"""
        Based on the following image labels and objects: {combined_labels}
        And the user's profile: {user_profile}
        Predict the most likely dish from this menu:
        {menu_text}
        If no exact match, suggest the closest dish and explain why it fits the labels and profile.
        Format the response as:
        **Dish**: [Dish Name]
        **Explanation**: [Reasoning]
        """
        dish_guess = gemini_model.generate_content(prompt).text.strip()
        st.success(f"Predicted Dish:\n{dish_guess}")

        # Feedback form
        with st.form("feedback_form"):
            feedback = st.radio("Is the predicted dish correct?", ["Yes", "No"])
            correct_dish = st.text_input("If No, what is the correct dish?", disabled=feedback == "Yes")
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                db.collection("dish_feedback").add({
                    "labels": combined_labels,
                    "predicted_dish": dish_guess,
                    "feedback": feedback,
                    "correct_dish": correct_dish if feedback == "No" else None,
                    "timestamp": time.time()
                })
                st.success("Feedback submitted!")

# TAB 2: Personalized Menu Recommendations
with tab2:
    st.header("Personalized AI Menu")
    menu = fetch_menu()
    menu_text = "\n".join([
        f"- {item['name']}: {item.get('description', '')} ({', '.join(item.get('dietary_tags', []))})"
        for item in menu
    ])
    user_profile = f"Diet: {', '.join(dietary) if dietary else 'None'}, Allergies: {', '.join(allergies) if allergies else 'None'}"
    prompt = f"Given user profile ({user_profile}) recommend 5 dishes:\n{menu_text}"
    ai_result = gemini_model.generate_content(prompt).text.strip()
    st.markdown(ai_result)

# TAB 3: Custom Filtering Options
with tab3:
    st.header("Custom Menu Filters")
    portion = st.selectbox("Portion Size", ["Regular", "Small", "Large"])
    ingredient_swap = st.text_input("Ingredient Swap")

    filtered_menu = []
    for item in menu:
        tags = item.get("dietary_tags", [])
        ingredients = item.get("ingredients", [])
        if (not dietary or any(d in tags for d in dietary)) and \
           (not allergies or all(a not in ingredients for a in allergies)):
            item_copy = item.copy()
            item_copy["portion_size"] = portion
            item_copy["ingredient_swap"] = ingredient_swap
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
            st.subheader(f"{entry['dish']} by {entry['staff']}")
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
    leaderboard = sorted(entries, key=lambda e: calculate_score(e), reverse=True)
    for i, entry in enumerate(leaderboard[:5]):
        st.write(f"**#{i+1} - {entry['dish']} by {entry['staff']} → {calculate_score(entry)} pts**")

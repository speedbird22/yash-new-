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
from fuzzywuzzy import fuzz

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

# TAB 1: AI Dish Detection (Enhanced)
with tab1:
    st.header("Visual Dish Detection (AI + Vision API)")
    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # Slightly higher contrast for clarity
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)  # Improve brightness
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        content = img_bytes.getvalue()

        # Vision API: Label detection, object localization, and text detection
        vision_image = vision.Image(content=content)
        label_response = vision_client.label_detection(image=vision_image)
        labels = [(label.description, label.score) for label in label_response.label_annotations if label.score > 0.7]
        obj_response = vision_client.object_localization(image=vision_image)
        objects = [(obj.name, obj.score) for obj in obj_response.localized_object_annotations]
        text_response = vision_client.text_detection(image=vision_image)
        texts = [text.description.lower().strip() for text in text_response.text_annotations[1:] if text.description.strip()]

        # Combine and filter detections
        combined_labels = [desc.lower() for desc, score in labels + objects]
        combined_labels = list(set(combined_labels + texts))
        st.write(f"Detected Labels, Objects, and Text: {combined_labels}")

        # Check if food-related
        food_related = any(
            label.lower() in ["food", "dish", "meal"] or "food" in label.lower() or any(food_term in label.lower() for food_term in ["pizza", "burger", "pasta", "salad", "sushi"])
            for label in combined_labels
        )
        if not food_related:
            st.warning("The image doesn't appear to contain food. Please upload a food-related image.")
            st.stop()

        # Cross-reference with Firestore menu
        menu = fetch_menu()
        menu_text = "\n".join([
            f"{item['name']}: {item.get('description', '')} (Ingredients: {', '.join(item.get('ingredients', []))}; Tags: {', '.join(item.get('dietary_tags', []))})"
            for item in menu
        ])
        user_profile = f"Diet: {', '.join(dietary) if dietary else 'None'}, Allergies: {', '.join(allergies) if allergies else 'None'}"

        # Calculate similarity scores for menu items
        matching_dishes = []
        for item in menu:
            item_text = ' '.join([
                item['name'].lower(),
                item.get('description', '').lower(),
                ' '.join(item.get('ingredients', [])).lower(),
                ' '.join(item.get('dietary_tags', [])).lower()
            ])
            score = max(fuzz.partial_ratio(label, item_text) for label in combined_labels)
            if score > 60:  # Threshold for relevance
                matching_dishes.append({
                    "name": item['name'],
                    "score": score,
                    "description": item.get('description', ''),
                    "ingredients": item.get('ingredients', []),
                    "dietary_tags": item.get('dietary_tags', []),
                    "id": item['id']
                })
        matching_dishes = sorted(matching_dishes, key=lambda x: x['score'], reverse=True)[:5]  # Top 5 matches

        # Gemini prompt for precise dish prediction
        prompt = f"""
        Analyze the following:
        - Image labels and objects: {labels + objects}
        - Detected text: {texts if texts else 'None'}
        - User profile: {user_profile}
        - Menu items: {menu_text}

        Tasks:
        1. Predict the most likely dish from the menu that matches the image, prioritizing high-confidence labels (score > 0.8) and detected text.
        2. If no exact match, suggest the closest dish and explain why it fits the labels, text, and user profile.
        3. Recommend 3 additional relevant dishes from the menu that align with the detected dish's characteristics and user preferences.

        Format the response as:
        **Predicted Dish**: [Dish Name]
        **Explanation**: [Reasoning]
        **Related Menu Items**:
        - [Dish Name]: [Description] (Similarity: [Score]%)
        - ...
        **Relevant Recommendations**:
        - [Dish Name]: [Reason]
        - ...
        """
        try:
            dish_guess = gemini_model.generate_content(prompt).text.strip()
            st.success(f"AI Dish Analysis:\n{dish_guess}")

            # Display related menu items as a table
            if matching_dishes:
                st.subheader("Related Menu Items")
                df = pd.DataFrame([
                    {
                        "Dish Name": dish['name'],
                        "Description": dish['description'],
                        "Ingredients": ', '.join(dish['ingredients']),
                        "Dietary Tags": ', '.join(dish['dietary_tags']),
                        "Similarity Score": f"{dish['score']}%"
                    }
                    for dish in matching_dishes
                ])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No closely related menu items found.")

        except Exception as e:
            st.error(f"AI analysis failed: {e}")

        # Feedback form
        with st.form("feedback_form"):
            feedback = st.radio("Is the predicted dish correct?", ["Yes", "No"])
            correct_dish = st.text_input("If No, what is the correct dish?", disabled=feedback == "Yes")
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                db.collection("dish_feedback").add({
                    "labels": combined_labels,
                    "texts": texts,
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

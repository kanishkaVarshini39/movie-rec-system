import streamlit as st
import requests

st.set_page_config(page_title="Netflix Matrix Demo Engine", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

# --- STATE SESSION INITIALIZATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.active_user_id = None
if 'current_view_movie' not in st.session_state:
    st.session_state.current_view_movie = None
# NEW: Initialize Watch History array
if 'watch_history' not in st.session_state:
    st.session_state.watch_history = []

# --- UI VISUAL VIEWS CONTROLLER ---

if not st.session_state.authenticated:
    st.title("🎬 Enterprise Content Recommendation Gateway")
    st.markdown("---")
    
    with st.container():
        user_input = st.text_input("Enter User ID (e.g., 30878):", placeholder="Enter numeric ID...")
        login_action = st.button("Initialize Personalized Pipeline", use_container_width=True)
        
        if login_action and user_input:
            if user_input.isdigit():
                st.session_state.authenticated = True
                st.session_state.active_user_id = int(user_input)
                st.rerun()
            else:
                st.error("Invalid Format. User ID must contain numbers only.")

else:
    # --- SIDEBAR (Profile & Watch History) ---
    st.sidebar.markdown(f"### 👤 Operational Profile\n**User ID:** `{st.session_state.active_user_id}`")
    if st.sidebar.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.active_user_id = None
        st.session_state.current_view_movie = None
        st.session_state.watch_history = [] # Clear history on logout
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕒 Session Watch History")
    if not st.session_state.watch_history:
        st.sidebar.caption("No movies rated yet this session.")
    else:
        # Display history in reverse order (newest first)
        for item in reversed(st.session_state.watch_history):
            st.sidebar.caption(f"**{item['Title']}** - {item['Rating']} ⭐")

    # --- MAIN VIEW A: GLOBAL DASHBOARD ---
    if st.session_state.current_view_movie is None:
        st.title(f"Target Queue Recommendations for User: {st.session_state.active_user_id}")
        st.write("_Displaying top 10 recommended movies based on your historical data model vectors._")
        
        try:
            response = requests.get(f"{BACKEND_URL}/recommendations/{st.session_state.active_user_id}")
            
            if response.status_code == 200:
                recommendations = response.json()
                ui_cols = st.columns(5)
                for rank_idx, item in enumerate(recommendations):
                    with ui_cols[rank_idx % 5]:
                        with st.container(border=True):
                            st.markdown(f"**{item['Title']}**")
                            st.caption(f"Year: {int(item['Year'])}")
                            
                            if st.button("Watch Trailer", key=f"rec_btn_{item['Movie_ID']}", use_container_width=True):
                                st.session_state.current_view_movie = item
                                st.rerun()
            else:
                st.error("Failed to generate predictive metrics.")
        except Exception as err:
            st.error(f"Network Connection Refused by API Service. Error: {err}")

    # --- MAIN VIEW B: DEEP DETAIL SCREEN ---
    else:
        selected_movie = st.session_state.current_view_movie
        
        if st.button("⬅️ Return to Global Recommendations Hub"):
            st.session_state.current_view_movie = None
            st.rerun()
            
        st.markdown("---")
        main_canvas_col, auxiliary_sidebar_col = st.columns([3, 1], gap="large")
        
        # LEFT PANEL (Main Content & Video)
        with main_canvas_col:
            st.title(f"Now Playing: {selected_movie['Title']}")
            st.caption(f"Metadata ID: {selected_movie['Movie_ID']} | Year: {int(selected_movie['Year'])}")
            
            st.video("https://vimeo.com/717840131") 
            
            st.markdown("### ⭐️ Rate this Movie")
            rating_score = st.slider("Select star rating:", min_value=1, max_value=5, value=3, step=1)
            
            if st.button("Commit Rating", use_container_width=True):
                payload_data = {
                    "user_id": int(st.session_state.active_user_id),
                    "movie_id": int(selected_movie['Movie_ID']),
                    "rating": int(rating_score)
                }
                try:
                    write_res = requests.post(f"{BACKEND_URL}/ratings", json=payload_data, timeout=30)
                    if write_res.status_code == 200:
                        # NEW: Add to history and show success notification
                        st.session_state.watch_history.append({
                            "Title": selected_movie['Title'],
                            "Rating": rating_score
                        })
                        st.success(f"Success! {rating_score} Stars committed. Added to Watch History.")
                    else:
                        st.error("Failed to persist data update transactions.")
                except Exception as ex:
                    st.error(f"Backend connection failure: {ex}")

        # RIGHT PANEL (More Like This)
        with auxiliary_sidebar_col:
            st.markdown("### 🍿 More Like This")
            st.write("_Similar movies based on latent vector analytics._")
            
            try:
                sim_res = requests.get(f"{BACKEND_URL}/movies/{selected_movie['Movie_ID']}/similar", timeout=30)
                
                if sim_res.status_code == 200:
                    related_items = sim_res.json()
                    
                    for sub_item in related_items:
                        with st.container(border=True):
                            st.markdown(f"**{sub_item['Title']}**")
                            st.caption(f"Year: {int(sub_item['Year'])}")
                            
                            unique_key = f"nav_{selected_movie['Movie_ID']}_to_{sub_item['Movie_ID']}"
                            
                            if st.button("Watch This", key=unique_key, use_container_width=True):
                                st.session_state.current_view_movie = sub_item
                                st.rerun()
                else:
                    st.caption("No similar movies found.")
            except Exception as loop_ex:
                st.caption(f"Error fetching similarity data: {loop_ex}")
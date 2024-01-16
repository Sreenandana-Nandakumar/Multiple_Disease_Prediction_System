import streamlit as st
import numpy as np
import keras
from PIL import Image
from streamlit_option_menu import option_menu

# Loading the trained models
try:
    malaria_model = keras.models.load_model('malaria.h5')
    pneumonia_model = keras.models.load_model('pneumoniaf.h5')
    brain_tumor_model = keras.models.load_model('braincancer.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# Function to preprocess the image before passing it to the model
def preprocess_image(image, model_type):
    if model_type == 'Malaria' or model_type == 'Pneumonia':
        img_size = (128, 128)
    elif model_type == 'Brain Tumor':
        img_size = (224, 224)
    else:
        raise ValueError("Invalid model type")

    img = image.resize(img_size)

    if img.mode == 'L':
        img = img.convert('RGB')

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to make a prediction using the whichever model the user selects
def predict_image(image, model_type):
    if model_type == 'Malaria':
        model = malaria_model
    elif model_type == 'Pneumonia':
        model = pneumonia_model
    elif model_type == 'Brain Tumor':
        model = brain_tumor_model
    else:
        raise ValueError("Invalid model type")

    processed_img = preprocess_image(image, model_type)
    prediction = model.predict(processed_img)
    return prediction


def login():
    st.title("Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    login_button = st.button("Login")

    if login_button:
        if authenticate(username, password):
            st.success("Successfully Logged in as Admin")
            st.session_state.logged_in = True
            st.experimental_rerun()
            if not st.session_state.logged_in:
                st.error("Error: Login state not persisted. Please try again.")
        else:
            st.error("Invalid credentials. Please try again.")
    st.markdown(
            """
            <style>
                body {
                    background: url("https://png.pngtree.com/background/20210710/original/pngtree-blue-technology-wind-medical-banner-picture-image_1034506.jpg") no-repeat center center fixed;
                    background-size: cover;
                    opacity: 0.875;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


def authenticate(username, password):
    return username == "admin" and password == "admin"


def main():
    if not st.session_state.get('logged_in', False):
        st.session_state.logged_in = login()

    if st.session_state.logged_in:
        st.title("Multiple Disease Prediction")

        with st.sidebar:
            selected = option_menu('Medical Image Classification',
                                ['Malaria Detection', 'Pneumonia Detection', 'Brain Tumor Detection'],
                                icons=['bug', 'lungs', 'clipboard-pulse'],
                                default_index=0)

        # Malaria Detection Interface
        if selected == 'Malaria Detection':
            uploaded_file = st.file_uploader("Choose a cell image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)

                try:
                    pred = predict_image(image, 'Malaria')
                    if pred[0][0] <= 0.5:
                        st.error("INFECTED CELL")
                    else:
                        st.success("UNINFECTED CELL")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        # Pneumonia Detection Interface
        elif selected == 'Pneumonia Detection':
            uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded X-ray Image.', use_column_width=True)

                try:
                    pred = predict_image(image, 'Pneumonia')
                    pneumonia_prob = pred[0][1]
                    normal_prob = pred[0][0]

                    if pneumonia_prob > normal_prob:
                        st.error("PNEUMONIA DETECTED")
                    else:
                        st.success("PNEUMONIA NOT DETECTED")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        # Brain Tumor Detection Interface
        elif selected == 'Brain Tumor Detection':
            uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Brain MRI Image.', use_column_width=True)

                try:
                    pred = predict_image(image, 'Brain Tumor')
                    if np.argmax(pred) == 0:
                        st.success("BRAIN TUMOR NOT DETECTED")
                    else:
                        st.error("BRAIN TUMOR DETECTED")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        st.markdown(
            """
            <style>
                body {
                    background: url('https://png.pngtree.com/background/20210710/original/pngtree-pink-medical-equipment-banner-background-picture-image_968645.jpg') no-repeat center center fixed;
                    background-size: cover;
                    opacity: 0.875;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()

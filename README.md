# Multiple_Disease_Prediction_System

This project presents a user-friendly interface tailored for medical professionals to forecast diseases like pneumonia, malaria, and brain tumors utilizing medical images. The interface enables users to effortlessly upload images of cells, X-rays, or MRI scans. Subsequently, the system harnesses deep learning models to scrutinize these images, furnishing predictions regarding the existence or non-existence of diseases.

### Features:
- Multi-disease Prediction: The interface supports the prediction of three diseases: malaria, pneumonia, and brain tumors, each with its own specialized deep learning model.
- User Authentication: Implemented a simple login system to restrict access to authorized users, ensuring data security and privacy.
- Interactive Interface: Utilized Streamlit to create an interactive and intuitive user interface, enabling easy navigation and seamless image upload.
- Real-time Prediction: Upon image upload, the system performs real-time analysis and provides immediate feedback on the presence or absence of diseases.
- Model Persistence: The trained deep learning models are loaded into the interface, allowing for quick and efficient predictions without the need for retraining.

### Technologies Used:
- Python
- Streamlit
- Keras (TensorFlow backend)
- PIL (Python Imaging Library)

### Installation and Usage:
- Clone the repository:
  ``` git clone https://github.com/Sreenandana-Nandakumar/Multiple_Disease_Prediction_System/ ```
- Running the application:
  ``` streamlit run app.py ```
- Navigate to the provided local URL in your web browser.
- Log in using your credentials (default: username - admin, password - admin).
- Select the type of medical image prediction from the sidebar.
- Upload the corresponding medical image (cell image for malaria, X-ray for pneumonia, MRI image for brain tumor).
- View the prediction result displayed on the interface.

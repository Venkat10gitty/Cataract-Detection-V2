import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import plotly.graph_objects as go

google_form_link = 'https://forms.gle/JVBEEKRGmKhCs8HZ6'

# Function to load the model
def load_model():
    model = torch.load('efficientnet_model.pth', map_location=torch.device('cpu'))
    model.eval()
    return model


# Function to process the image
def process_image(image):
    # Define the same transforms as used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Function to predict the class
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted.item()].item()
        return predicted.item(), confidence

# Function to plot confidence pie chart
def plot_confidence(confidence):
    labels = ['Cataract', 'Normal']
    sizes = [confidence, 1-confidence]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    return fig

# Streamlit App UI
st.title('Cataract Detection App')

model = load_model()
st.header('Upload an eye image')
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Eye Image', use_column_width=True)

    # Cropping functionality
    st.write("Select the area to crop:")

    # Horizontal crop sliders
    col1, col2 = st.columns(2)
    with col1:
        x1, x2 = st.slider("Select horizontal range for cropping", 0, image.width, (0, image.width))

    # Vertical crop sliders
    with col2:
        y1, y2 = st.slider("Select vertical range for cropping", 0, image.height, (0, image.height))

    # Apply cropping
    cropped_image = image.crop((x1, y1, x2, y2))

    # Display cropped image
    st.image(cropped_image, caption='Cropped Image', use_column_width=True)
    
    processed_image = process_image(image)
    prediction, confidence = predict(model, processed_image)
    if st.button('Detect'):
        if(confidence > 0.7):
            if prediction == 0:
                st.markdown(f"<h2 style='color: red;'>Cataract Detected </h2>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Pie(labels=['Cataract', 'No Cataract'], 
                                            values=[confidence, 1 - confidence],
                                            hoverinfo='label+percent', 
                                            pull=[0, 0])])
                fig.update_layout(title_text='Cataract Detection Probability')
                st.plotly_chart(fig)
            else:
                st.markdown(f"<h2 style='color: green;'>No Cataract Detected </h2>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Pie(labels=['Cataract', 'No Cataract'], 
                                            values=[1-confidence, confidence],
                                            hoverinfo='label+percent', 
                                            pull=[0, 0])])
                fig.update_layout(title_text='Cataract Detection Probability')
                st.plotly_chart(fig)    
            st.subheader("Doctor's Verification 🩺")
            st.markdown(f"[Click here to provide feedback on the cataract detection results]({google_form_link})", unsafe_allow_html=True)
        else:
            st.error("Sorry, uploaded image is not clear.")
        


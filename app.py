import streamlit as st
from sklearn import datasets
import joblib
import os
st.set_page_config(layout="wide")

files = os.listdir(os.getcwd())
st.write("Files in the current directory:")
for file in files:
    st.write(file)


iris = datasets.load_iris()
mapping_dict = dict(zip(range(3), iris.target_names))  

# ---------- Start of Streamlit app
file_path = os.path.join(os.getcwd(), 'iris_model.pkl')

# Load model
st.write(file_path)
st.write(os.getcwd())
loaded_model = joblib.load(file_path)
# loaded_model = joblib.load('./iris_model.pkl')


st.markdown("<h1 style='text-align: center; color: black;'>FLOWER CLASSIFICATION </h1>",
            unsafe_allow_html=True)
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# --------- User input
st.markdown("<h3 style='text-align: center; color: black;'>USER INPUT </h3>",
            unsafe_allow_html=True
)
col1, col2 = st.columns([1,1])

# -------- User inputs in columns 1 and 2 
# Sepal Length, Sepal Width, Petal Length and Petal Width
sepal_length = col1.number_input(
    "Sepal Length (cm)",
    value=None,
    min_value=4.3,
    max_value=7.9,
    placeholder="Enter sepal length in cm...", 
    format="%d",
    label_visibility="visible",
)

sepal_width = col1.number_input(
    "Sepal Width (cm)",
    value=None,
    min_value=2.0,
    max_value=4.4,
    placeholder="Enter sepal width in cm...", 
    format="%d",
    label_visibility="visible",
)

petal_length = col2.number_input(
    "Petal Length (cm)",
    value=None,
    min_value=1.0,
    max_value=6.9,
    placeholder="Enter petal length in cm...", 
    format="%d",
    label_visibility="visible",
)

petal_width = col2.number_input(
    "Petal Width (cm)",
    value=None,
    min_value=0.1,
    max_value=2.5,
    placeholder="Enter petal width in cm...", 
    format="%d",
    label_visibility="visible",
)

submit = st.button("Predict", help="Click here to start prediction", 
                    type="primary", use_container_width=True,
)
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# ------- Prediction section 
st.markdown(
    "<h1 style='text-align: center; color: black;'>PREDICTION </h1>",
    unsafe_allow_html=True
)

files = os.listdir(os.getcwd())
st.write("Files in the current directory:")
for file in files:
    st.write(file)

if submit == True: 
    # Create a dictionary with the user's input values  
    user_input = {  
        "sepal_length": sepal_length,  
        "sepal_width": sepal_width,  
        "petal_length": petal_length,  
        "petal_width": petal_width,  
    }  
  
    # Convert the dictionary to a JSON object  
#     user_input_json = json.dumps(user_input)  
  
    # st.write(user_input_json)   # Write to streamlit the input json if needed

    # ---------------- TODO -----------------------
    # Pass the user_input_json to your trained model in the backend
    # Retrieve the prediction_json from the backend
    # Apply the mapping defined at the beginning to print the name of iris flower species

#     prediction_json = {"prediction" : 1}  # dummy
#     flower_name = mapping_dict[prediction_json['prediction']]
    user_input_json = [list(user_input.values())]
    flower_name = mapping_dict[loaded_model.predict(user_input_json)[0]]

    st.metric("Flower type", 
            str(flower_name),
            help="Flower type predicted by trained model", 
            label_visibility="visible",
            )  

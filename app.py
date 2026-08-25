import streamlit as st
import joblib

from preprocess import transform_text

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)


st.title("Email/SMS Spam Classifier")

st.write(
    "Enter an SMS message below and the model will "
    "predict whether it is Spam or Ham."
)


message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You won a free prize..."
)


if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Preprocess
        transformed_text = transform_text(message)

        # TF-IDF
        vectorized_text = tfidf.transform(
            [transformed_text]
        )

        # Prediction
        prediction = model.predict(
            vectorized_text
        )[0]

        # Probability
        probability = model.predict_proba(
            vectorized_text
        )[0]

        spam_probability = probability[1]

        if prediction == 1:

            st.error("🚨 SPAM")

            st.write(
                f"Spam probability: "
                f"{spam_probability * 100:.2f}%"
            )

        else:

            st.success("Not Spam")

            st.write(
                f"Spam probability: "
                f"{spam_probability * 100:.2f}%"
            )
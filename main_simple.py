# 1. Importing necessary libraries
import streamlit as st  # Import the Streamlit library
import random  # Import the random library
import time  # Import the time library

# 2. Creating a title for our streamlit web application
st.title("Simple GenAI Assistant")  # Set the title of the web application

# 3. Initializing the chat history in the session state (how the chatbot tracks things)
if "messages" not in st.session_state:  # Check if "messages" exists in session state
    st.session_state.messages = []  # Initialize "messages" as an empty list

# 4. Displaying the existing chat messages from the user and the chatbot
for message in st.session_state.messages:  # For every message in the chat history
    with st.chat_message(message["role"]):  # Create a chat message box
        st.markdown(message["content"])  # Display the content of the message
        print(message["content"])

# 5. Accepting the user input and adding it to the message history
if prompt := st.chat_input("What is up?"):  # If user enters a message
    with st.chat_message("user"):  # Display user's message in a chat message box
        st.markdown(prompt)  # Display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})  # Add user's message to chat history

# 6. Generating and displaying the assistant's response
with st.chat_message("assistant"):  # Create a chat message box for the assistant's response
    message_placeholder = st.empty()  # Create an empty placeholder for the assistant's message
    full_response = ""  # Initialize an empty string for the full response
    assistant_response = random.choice([
        "Hello there! How can I assist you today?",
        "Hi, human! Is there anything I can help you with?",
        "Do you need help?",
        "Greetings! What can I do for you today?"  # Added new response option
    ])  # Select assistant's response randomly

    # Simulate "typing" effect by gradually revealing the response
    for chunk in assistant_response.split():  # For each word in the response
        full_response += chunk + " "
        time.sleep(0.05)  # Small delay between each word
        message_placeholder.markdown(full_response + "▌")  # Update placeholder with current full response and a blinking cursor

    message_placeholder.markdown(full_response)  # Remove cursor and display full response
    st.session_state.messages.append({"role": "assistant", "content": full_response})  # Add assistant's response to chat history
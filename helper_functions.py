# Third-Party Imports
import streamlit as st
import langchain
from index_functions import load_data
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.model.query_response import QueryResponse
from pinecone_text.sparse import BM25Encoder
from index_functions import produce_embeddings
from index_functions import issue_hybrid_query

INDEX_NAME = "hybrid-search"

# Main function to generate responses from OpenAI's API, not considering indexed data
def generate_response(prompt, history, model_name, openai_client, temperature):
    # Fetching the last message sent by the chatbot from the conversation history
    chatbot_message = history[-1]['content']

    # Fetching the first message that the user sent from the conversation history
    first_message = history[1]['content']

    # Constructing a comprehensive prompt to feed to OpenAI for generating a response
    full_prompt = f"{prompt}\n\
    ### The original message: {first_message}. \n\
    ### Your latest message to me: {chatbot_message}. \n\
    ### Previous conversation history for context: {history}"

    # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
    api_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # Assuming there's a need to include any system or user messages for context
            # {"role": "system", "content": "Your system message here"},
            {"role": "user", "content": full_prompt}
        ],
        temperature=temperature,
        max_tokens=350,  # Assuming a max_tokens value. Adjust as necessary.
        n=1,  # Number of completions to generate
    )

    # Extracting the generated response content from the API response object
    full_response = api_response.choices[0].message.content.strip()

    # Yielding a response object containing the type and content of the generated message
    yield {"type": "response", "content": full_response}

# Similar to generate_response but also includes indexed data to provide more context-aware and data-driven responses
def generate_response_index(prompt, history, model_name, temperature):

    # Retrive data from knowledge store in Pinecone
    # Let's grab the textual metadata from our search results:
    # Query Our Hybrid Docs
    # Fetching the last message from the user 
    chatbot_message = history[-1]['content']

    bm25 = st.session_state.session_state['bm25']
    chat_engine = st.session_state.session_state['openai_client']
    pinecone = st.session_state.session_state['pinecone']
    index = pinecone.Index(INDEX_NAME)

    query_sembedding = bm25.encode_queries(chatbot_message)
    query_dembedding = produce_embeddings([chatbot_message])
    # Note, for our dense embedding (`query_dembedding`), we need to grab the 1st value [0] since Pinecone expects a Numpy array when queried:
    # when you get further down the results list, you'll see that we get an equation we can use to calculate KNN. That's a bit more useful than #3 in our pure keyword search, which is a bibliography entry. 
    hybrid_3 = issue_hybrid_query(index, query_sembedding, query_dembedding[0], 0.1, 7)
    hybrid_context = [i.get('metadata').get('text') for i in hybrid_3.get('matches')]
    #pure_keyword_context = [i.get('metadata').get('text') for i in pure_keyword.get('matches')]
    #pure_semantic_context = [i.get('metadata').get('text') for i in pure_semantic.get('matches')]

    # We are then going to combine this "context" with our original query in a format that our LLM likes:

    hybrid_augmented_query = "\n\n---\n\n".join(hybrid_context)+"\n\n-----\n\n"+chatbot_message
    #pure_keyword_augmented_query = "\n\n---\n\n".join(pure_keyword_context)+"\n\n-----\n\n"+chatbot_message
    #pure_semantic_augmented_query = "\n\n---\n\n".join(pure_keyword_context)+"\n\n-----\n\n"+chatbot_message
    # Adding the indexed data to the prompt to make the chatbot response more context-aware and data-driven

    # We are then going to give our LLM some instructions for how to act:
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know".
    """

    print(hybrid_augmented_query)

    # TODO Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
    api_response = chat_engine.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": hybrid_augmented_query}
        ],
        temperature=temperature,
        max_tokens=350,  # Assuming a max_tokens value. Adjust as necessary.
        n=1,  # Number of completions to generate
    )
    
    # Extracting the generated response content from the API response object
    full_response = api_response.choices[0].message.content
    
    # Yielding a response object containing the type and content of the generated message
    yield {"type": "response", "content": full_response}

#################################################################################
# Additional, specific functions I had in the Innovation CoPilot for inspiration:

# Function returns a random thanks phrase to be used as part of the CoPilots reply
# Note: Requires a dictionary of 'thanks phrases' to work properly
def get_thanks_phrase():
    selected_phrase = random.choice(thanks_phrases)
    return selected_phrase

# Function to randomize initial message of CoPilot
# Note: Requires a dictionary of 'initial messages' to work properly
def get_initial_message():
    initial_message = random.choice(initial_message_phrases)
    return initial_message

# Function to generate the summary; used in part of the response
def generate_summary(model_name, temperature, summary_prompt):
    summary_response = client.ChatCompletion.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are an expert at summarizing information effectively and making others feel understood"},
            {"role": "user", "content": summary_prompt}
        ]
    )
    summary = summary_response.choices[0].message.content
    print(f"summary: {summary}, model name: {model_name}, temperature: {temperature})")
    return summary

# Function used to enable 'summary' mode in which the CoPilot only responds with bullet points rather than paragraphs
def transform_bullets(content):
    try:
        prompt = f"Summarize the following content in 3 brief bullet points while retaining the structure and conversational tone (using wording like 'you' and 'your idea'):\n{content}"
        response = client.ChatCompletion.create(
        model="gpt-4",
        temperature=.2,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(response)
        print("Error in transform_bullets:", e)
        return content  # Return the original content as a fallback

# Function to add relevant stage specific context into prompt
def get_stage_prompt(stage):
      #Implementation dependent on your chatbots context
      return

# Function to grade the response based on length, relevancy, and depth of response
def grade_response(user_input, assistant_message, idea):
      #Implementation dependent on your chatbots context
      return      

# Function used to generate a final 'report' at the end of the conversation, summarizing the convo and providing a final recomendation
def generate_final_report():
      #Implementation dependent on your chatbots context
      return

#use instructor py library and write a chat completions function that takes a user input and gives output using the gpt-4o model 

import openai

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

def get_chat_completion(system_prompt, user_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    # Define your system and user prompts
    system_prompt = "You are a helpful assistant."
    user_prompt = "Can you explain how photosynthesis works?"

    # Get the completion from the chat model
    completion = get_chat_completion(system_prompt, user_prompt)
    
    # Print the completion
    print("Assistant:", completion)

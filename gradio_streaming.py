import gradio as gr
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")
genai.configure(api_key=GOOGLE_API_KEY)

# Create the generative model
model = genai.GenerativeModel('gemini-pro')

async def generate_word_streaming_response(prompt):
    """
    Generate a streaming response from the Gemini API, word by word.
    
    Args:
        prompt (str): The user's input text
    
    Yields:
        str: Incremental words of the generated response
    """
    try:
        # Use streaming generation
        response = model.generate_content(prompt, stream=True)
        
        # Buffer to accumulate partial words
        current_response = ""
        
        for chunk in response:
            if chunk.parts:
                # Split the new chunk into words
                new_text = chunk.text
                words = new_text.split()
                
                # Yield words incrementally
                for word in words:
                    current_response += word + " "
                    yield current_response.strip()
                    await asyncio.sleep(0.1)  # Small delay to simulate word-by-word streaming
        
        # Ensure final response is fully displayed
        if current_response:
            yield current_response.strip()
    
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def create_gemini_streaming_interface():
    """
    Create a Gradio interface with word-by-word streaming output.
    
    Returns:
        gr.Blocks: Configured Gradio application
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Gemini AI Word-by-Word Streaming Chatbot")
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_text = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Type your question or request here..."
                )
                submit_btn = gr.Button("Generate Response")
                
                # Output components
                output_text = gr.Textbox(
                    label="Gemini Response",
                    placeholder="Response will appear here...",
                    lines=10
                )
                
                # Example prompts
                examples = gr.Examples(
                    examples=[
                        ["Write a detailed explanation of how neural networks work"],
                        ["Compose a creative short story about artificial intelligence"],
                        ["Describe the potential future impacts of quantum computing"]
                    ],
                    inputs=[input_text],
                    outputs=[output_text]
                )
                
                # Bind the submit button to the streaming response function
                submit_btn.click(
                    fn=generate_word_streaming_response,
                    inputs=[input_text],
                    outputs=[output_text],
                    api_name="generate"
                )
        
        return demo

# Launch the Gradio app
def main():
    app = create_gemini_streaming_interface()
    app.launch(
        share=False  # Set to True if you want a public link
    )

if __name__ == "__main__":
    main()
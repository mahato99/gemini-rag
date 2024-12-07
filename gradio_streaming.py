import gradio as gr
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Create the generative model
model = genai.GenerativeModel('gemini-pro')

def generate_streaming_response(prompt):
    """
    Generate a streaming response from the Gemini API.
    
    Args:
        prompt (str): The user's input text
    
    Yields:
        str: Incremental parts of the generated response
    """
    try:
        # Use streaming generation
        response = model.generate_content(prompt, stream=True)
        
        # Accumulate and yield partial responses
        current_response = ""
        for chunk in response:
            if chunk.parts:
                current_response += chunk.text
                yield current_response
    
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def create_gemini_streaming_interface():
    """
    Create a Gradio interface with streaming output.
    
    Returns:
        gr.Blocks: Configured Gradio application
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Gemini AI Streaming Chatbot")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Enter your prompt", 
                    placeholder="Type your question or request here..."
                )
                submit_btn = gr.Button("Generate Response")
            
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
            fn=generate_streaming_response, 
            inputs=[input_text], 
            outputs=[output_text],
            api_name="generate"
        )
    
    return demo

# Launch the Gradio app
def main():
    app = create_gemini_streaming_interface()
    app.launch()

if __name__ == "__main__":
    main()
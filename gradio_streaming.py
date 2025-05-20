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
model = genai.GenerativeModel('gemini-1.5-pro-latest')

async def generate_word_streaming_response(prompt):
    """
    Generate a streaming response from the Gemini API, yielding accumulated text chunks.
    
    Args:
        prompt (str): The user's input text
    
    Yields:
        str: Accumulated parts of the generated response
    """
    try:
        # Use streaming generation
        response_stream = model.generate_content(prompt, stream=True)
        
        full_response = ""
        for chunk in response_stream:
            # Standard way to get text from Gemini chunk
            if hasattr(chunk, 'text') and chunk.text:
                full_response += chunk.text
                yield full_response
            # Fallback if chunk.text is not present but parts are (less common for simple text)
            elif hasattr(chunk, 'parts') and chunk.parts:
                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text:
                        full_response += part.text
                        yield full_response
        # The loop should handle yielding the complete response progressively.
        # A final yield might be redundant if all chunks are processed and yielded.
    
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


# Gradio interface
interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Question"),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Context"),
        gr.Textbox(label="Confidence Score")
    ],
    title="Enterprise QA System",
    description="Ask questions about general knowledge topics"
)

# Launch in Colab
interface.launch(debug=True)


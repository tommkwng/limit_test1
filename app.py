import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("AI Story Generator with Large Models")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    # Display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Your Image", width=300)
    
    if st.button("Generate Story"):
        # LARGER Image-to-Text model (BLIP-2 is much larger and better)
        with st.spinner("Analyzing image with BLIP-2 (large model)..."):
            image_to_text = pipeline(
                "image-to-text", 
                model="Salesforce/blip2-opt-2.7b",  # 2.7B parameter model
                max_new_tokens=50
            )
            description = image_to_text(image)[0]['generated_text']
        
        st.write(f"**Detailed image description:** {description}")
        
        # LARGER Text Generation model (Llama 2 7B)
        with st.spinner("Creating story with Llama 2 (7B model)..."):
            text_generation = pipeline(
                "text-generation", 
                model="meta-llama/Llama-2-7b-chat-hf",  # 7B parameter model
                torch_dtype="auto",
                device_map="auto"
            )
            
            prompt = f"""You are a creative children's storyteller. 
            Based on this scene: {description}
            Write a magical, engaging story for children aged 3-8.
            Keep it under 150 words."""
            
            story = text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )[0]['generated_text']
        
        st.write(f"**Story:** {story}")
        
        # Text to Speech (using a larger TTS model)
        with st.spinner("Generating audio..."):
            text_to_speech = pipeline(
                "text-to-speech", 
                model="microsoft/speecht5_tts",  # Larger TTS model
            )
            audio = text_to_speech(story[:500])  # Limit text length
            
            # Play audio
            st.audio(audio["audio"], sample_rate=audio["sampling_rate"])

st.info("Note: These are large models (10GB+ total). First run will download them.")

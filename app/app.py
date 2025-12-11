import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datetime import datetime
import time
import textwrap
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .original-box {
        background-color: #fffacd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffa500;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_model():
    """Load the T5 model and tokenizer"""
    model_path = "BrianAlex1/t5-summarizer-news"
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        # Set to evaluation mode
        model.eval()
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def summarize_text(text, model, tokenizer, device, max_length=150, min_length=30, num_beams=4):
    """Summarize text using the T5 model"""
    try:
        # Prepare input
        input_text = f"summarize: {text}"
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            input_text,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Generate summary
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
        
        # Decode summary
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_compression_ratio(original, summary):
    """Calculate compression ratio"""
    original_words = len(original.split())
    summary_words = len(summary.split())
    if original_words == 0:
        return 0
    return round((1 - summary_words / original_words) * 100, 2)

def count_sentences(text):
    """Count sentences in text"""
    return len([s for s in text.split('.') if s.strip()])

# Sample texts for quick testing - CNN Daily Mail News Articles
SAMPLE_TEXTS = {
    "Airplane Seat Safety": """Ever noticed how plane seats appear to be getting smaller and smaller? With increasing numbers of people taking to the skies, some experts are questioning if having such packed out planes is putting passengers at risk. They say that the shrinking space on aeroplanes is not only uncomfortable - it's putting our health and safety in danger. More than squabbling over the arm rest, shrinking space on planes putting our health and safety in danger. This week, a U.S consumer advisory group set up by the Department of Transportation said at a public hearing that while the government is happy to set standards for animals flying on planes, it doesn't stipulate a minimum amount of space for humans. 'In a world where animals have more rights to space and food than humans,' said Charlie Leocha, consumer representative on the committee. 'It is time that the DOT and FAA take a stand for humane treatment of passengers.' Tests conducted by the FAA use planes with a 31 inch pitch, a standard which on some airlines has decreased. Many economy seats on United Airlines have 30 inches of room, while some airlines offer as little as 28 inches. The distance between two seats from one point on a seat to the same point on the seat behind it is known as the pitch.""",
    
    "Zoo Incident": """A drunk teenage boy had to be rescued by security after jumping into a lions' enclosure at a zoo in western India. Rahul Kumar, 17, clambered over the enclosure fence at the Kamla Nehru Zoological Park in Ahmedabad, and began running towards the animals, shouting he would 'kill them'. Mr Kumar explained afterwards that he was drunk and 'thought I'd stand a good chance' against the predators. Mr Kumar had been sitting near the enclosure when he suddenly made a dash for the lions, surprising zoo security. The intoxicated teenager ran towards the lions, shouting: 'Today I kill a lion or a lion kills me!' A zoo spokesman said: 'Guards had earlier spotted him close to the enclosure but had no idea he was planing to enter it. Fortunately, there are eight moats to cross before getting to where the lions usually are and he fell into the second one, allowing guards to catch up with him and take him out. We then handed him over to the police.' Kumar later explained: 'I don't really know why I did it. I was drunk and thought I'd stand a good chance.' A police spokesman said: 'He has been cautioned and will be sent for psychiatric evaluation.'""",
    
    "News Article 3": """Breaking news and important developments shape our world every day. Media outlets across the globe work tirelessly to bring accurate reporting to millions of readers. From international events to local stories, journalism plays a crucial role in keeping the public informed and educated. Professional journalists follow strict ethical guidelines to ensure factual accuracy and balanced reporting. News organizations invest in investigative journalism to uncover truths and hold institutions accountable. The digital age has transformed how news is consumed, with social media platforms becoming primary sources for many readers. Quality journalism requires resources, dedication, and commitment to truth in service of the public interest.""",
    
    "News Article 4": """Major developments continue to unfold across various sectors of the economy and society. Businesses innovate to meet changing consumer demands and market conditions. Financial markets respond to global economic indicators and policy decisions. Technological advancements create new opportunities and challenges for industries worldwide. Employment trends reflect shifts in consumer behavior and technological disruption. Corporate leaders navigate complex regulatory environments while pursuing growth strategies. Stakeholders from different sectors collaborate to address shared challenges. Industry analysts forecast future trends based on current market data and consumer insights. Economic growth depends on stable policies, skilled workforce, and technological innovation.""",
}

# Main title and header
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #667eea; font-size: 3em;'>📝 AI Text Summarizer</h1>
        <p style='font-size: 1.2em; color: #666;'>Transform lengthy texts into concise, meaningful summaries</p>
    </div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading AI model... This might take a moment on first load"):
    model, tokenizer, device = load_model()

if model is None or tokenizer is None:
    st.error("Failed to load the model. Please check the model files.")
    st.stop()

# Display device info
device_info = f"🚀 Running on: {device.upper()}"
col1, col2, col3 = st.columns(3)
with col1:
    st.info(device_info)
with col2:
    st.info(f"📊 Model: T5 (Summarization)")
with col3:
    st.info(f"🔧 Device Memory: Optimized")

st.divider()

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["📝 Summarize", "🎯 Sample Texts", "⚙️ Settings", "ℹ️ About"])

with tab1:
    st.header("Text Summarization")
    st.write("Enter your text below and let our AI create a concise summary for you!")
    
    # Input options
    col1, col2 = st.columns([3, 1])
    with col1:
        input_method = st.radio("Choose input method:", ["📄 Paste Text", "📋 Sample Text"], horizontal=True)
    
    with col2:
        st.write("")
        st.write("")
    
    # Input text
    if input_method == "📄 Paste Text":
        input_text = st.text_area(
            "Enter the text you want to summarize:",
            height=250,
            placeholder="Paste your text here...",
            label_visibility="collapsed"
        )
    else:
        sample_choice = st.selectbox("Choose a sample text:", list(SAMPLE_TEXTS.keys()))
        input_text = SAMPLE_TEXTS[sample_choice]
        st.text_area(
            "Sample text:",
            value=input_text,
            height=250,
            disabled=True,
            label_visibility="collapsed"
        )
    
    if input_text:
        # Advanced settings in expander
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                max_length = st.slider("Max Summary Length (words):", 30, 500, 150, step=10)
            with col2:
                min_length = st.slider("Min Summary Length (words):", 10, 200, 30, step=5)
            with col3:
                num_beams = st.slider("Beam Search Width:", 1, 8, 4, step=1)
        
        # Summarize button
        if st.button("✨ Generate Summary", key="summarize_btn", use_container_width=True):
            with st.spinner("🤖 AI is working on your summary..."):
                start_time = time.time()
                summary = summarize_text(
                    input_text,
                    model,
                    tokenizer,
                    device,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams
                )
                end_time = time.time()
            
            # Display results
            st.success(f"✅ Summary generated in {end_time - start_time:.2f} seconds!")
            
            # Create columns for results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📄 Original Text")
                with st.container():
                    st.markdown(f'<div class="original-box">{input_text}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ✨ Summary")
                with st.container():
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
            
            # Statistics
            st.divider()
            st.markdown("### 📊 Summary Statistics")
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            original_words = len(input_text.split())
            summary_words = len(summary.split())
            compression = calculate_compression_ratio(input_text, summary)
            
            with stats_col1:
                st.metric("📝 Original Words", original_words)
            with stats_col2:
                st.metric("✂️ Summary Words", summary_words)
            with stats_col3:
                st.metric("📉 Compression Ratio", f"{compression}%")
            with stats_col4:
                st.metric("⚡ Processing Time", f"{end_time - start_time:.2f}s")
            
            # Additional metrics
            st.divider()
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            original_sentences = count_sentences(input_text)
            summary_sentences = count_sentences(summary)
            
            with metrics_col1:
                st.metric("Sentences (Original)", original_sentences)
            with metrics_col2:
                st.metric("Sentences (Summary)", summary_sentences)
            with metrics_col3:
                avg_word_length_original = round(len(input_text.replace(" ", "")) / original_words, 2)
                st.metric("Avg Word Length (Original)", avg_word_length_original)
            
            # Copy to clipboard
            st.divider()
            st.markdown("### 📋 Export Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.text_area("Copy summary:", value=summary, height=150, disabled=True)
            
            with col2:
                # Download as text file
                st.download_button(
                    label="📥 Download Summary (.txt)",
                    data=summary,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                # Download both original and summary
                combined_text = f"ORIGINAL TEXT:\n{'-'*50}\n{input_text}\n\nSUMMARY:\n{'-'*50}\n{summary}"
                st.download_button(
                    label="📥 Download Both (.txt)",
                    data=combined_text,
                    file_name=f"summary_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("👉 Enter or select some text to get started!")

with tab2:
    st.header("Sample Texts & Quick Demo")
    st.write("Click on any of the sample texts to quickly test the summarizer!")
    
    for idx, (title, sample) in enumerate(SAMPLE_TEXTS.items()):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"📚 {title}")
                st.write(sample[:200] + "...")
                word_count = len(sample.split())
                st.caption(f"📝 {word_count} words | 📖 {count_sentences(sample)} sentences")
            
            with col2:
                if st.button(f"Try it →", key=f"demo_{idx}", use_container_width=True):
                    st.session_state.demo_text = sample
                    st.session_state.demo_title = title
    
    if "demo_text" in st.session_state:
        st.divider()
        st.markdown(f"### ✨ Summarizing: {st.session_state.demo_title}")
        
        with st.spinner("🤖 Generating summary..."):
            start_time = time.time()
            summary = summarize_text(
                st.session_state.demo_text,
                model,
                tokenizer,
                device,
                max_length=150,
                min_length=30,
                num_beams=4
            )
            end_time = time.time()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Original:**")
            st.info(st.session_state.demo_text)
        
        with col2:
            st.markdown("**Summary:**")
            st.success(summary)
        
        # Show statistics
        compression = calculate_compression_ratio(st.session_state.demo_text, summary)
        m1, m2, m3 = st.columns(3)
        m1.metric("Compression", f"{compression}%")
        m2.metric("Time", f"{end_time - start_time:.2f}s")
        m3.metric("Reduction", f"{len(st.session_state.demo_text.split()) - len(summary.split())} words")

with tab3:
    st.header("⚙️ Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.write("""
        - **Model Type:** T5 (Text-To-Text Transfer Transformer)
        - **Task:** Text Summarization
        - **Model Size:** 768 hidden dimensions
        - **Layers:** 12 encoder + 12 decoder layers
        - **Heads:** 12 attention heads
        - **Vocabulary:** 32,128 tokens
        - **Max Sequence Length:** 512 tokens
        """)
    
    with col2:
        st.subheader("System Information")
        st.write(f"""
        - **Processing Device:** {device.upper()}
        - **PyTorch Version:** {torch.__version__}
        - **CUDA Available:** {'Yes' if torch.cuda.is_available() else 'No'}
        """)
        if torch.cuda.is_available():
            st.write(f"- **GPU Name:** {torch.cuda.get_device_name(0)}")
    
    st.divider()
    
    st.subheader("Default Summarization Parameters")
    st.info("""
    **Beam Search:** Uses 4 parallel beams for better quality summaries
    
    **Length Penalty:** 2.0 to encourage longer, more informative summaries
    
    **No Repeat N-grams:** Size 3 to prevent repetition in the output
    
    **Early Stopping:** Enabled to stop generation once all beams reach the end token
    """)

with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 What is Text Summarization?
    Text summarization is the process of automatically generating a shorter version of a text document 
    that captures the most important information. This application uses advanced AI to create abstractive 
    summaries (generating new text) rather than simply extracting key sentences.
    
    ### 🧠 How It Works
    This application uses the **T5 (Text-To-Text Transfer Transformer)** model, a state-of-the-art 
    language model pre-trained on multiple NLP tasks including summarization. The model has been 
    fine-tuned to:
    - Understand context and identify key information
    - Generate coherent and concise summaries
    - Maintain the essence of the original text
    
    ### ✨ Key Features
    - **Interactive UI**: User-friendly interface built with Streamlit
    - **Real-time Processing**: Get summaries instantly
    - **Customizable Parameters**: Adjust summary length and beam search settings
    - **Advanced Statistics**: View compression ratios and processing metrics
    - **Sample Texts**: Quick demo with pre-loaded examples
    - **Export Options**: Download summaries in multiple formats
    - **GPU Acceleration**: Automatic GPU support for faster processing
    
    ### 🚀 Use Cases
    - **Academic Research**: Summarize research papers and articles
    - **News & Media**: Quickly understand news stories
    - **Business**: Extract key points from reports and documents
    - **Content Creation**: Generate summaries for social media
    - **Learning**: Get concise summaries of educational material
    
    ### 📚 About the Model
    The T5 model (Text-To-Text Transfer Transformer) is developed by Google and treats all NLP tasks 
    as a text-to-text problem. It has been trained on diverse tasks and shows excellent performance 
    on summarization benchmarks.
    
    ### 🛠️ Technology Stack
    - **Framework**: Streamlit for web interface
    - **Model**: Hugging Face Transformers
    - **Backend**: PyTorch
    - **Language**: Python 3.8+
    
    ---
    *Built with 💙 for Deep Learning Project | Semester 5*
    """)
    
    # Display current timestamp
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

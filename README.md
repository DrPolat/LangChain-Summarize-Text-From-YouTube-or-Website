# LangChain-Summarize-Text-From-YouTube-or-Website

This Streamlit-based application allows users to summarize content from YouTube videos or websites using the Groq API and LangChain framework.

## Features:
Groq API Integration: Utilizes the Groq API for large language model inference.
Content Summarization: Provides a summarized output from both YouTube videos and websites.
Validation & Input Handling: Includes validation for API key and URL inputs, ensuring a smooth user experience.
Document Processing: Uses loaders for YouTube and website content, followed by text splitting and summarization.

## Methodology:
Input Handling: Users input their Groq API key and a URL (either YouTube or a website).
Data Loading: Depending on the URL type, either YoutubeLoader or UnstructuredURLLoader is used.
Text Splitting: Documents are split into smaller chunks for summarization using RecursiveCharacterTextSplitter.

##  Summarization: The LangChain framework, specifically the map_reduce chain, is employed to generate a comprehensive summary from the split chunks.
The application is designed to provide a seamless experience for summarizing lengthy online content.

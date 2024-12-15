import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("Summarize Text From Youtube or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 1000 words:
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])
if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False,
                                                          language=["en","tr"],
                                                          translation="en",)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                final_document = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100).split_documents(docs)

                chunks_prompt="""
                Please summarize the below speech:
                Speech:`{text}'
                Summary:
                """
                map_prompt_template=PromptTemplate(input_variables=['text'],
                                                    template=chunks_prompt)
                final_prompt='''
                Provide the final summary of the entire speech .
                Have an introduction, development and conclusion sections
                Speech:{text}

                '''
                final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

                ## Chain For Summarization
                summary_chain=load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt_template,
                    combine_prompt=final_prompt_template,
                    verbose=True)
                output_summary=summary_chain.run(final_document)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
                    
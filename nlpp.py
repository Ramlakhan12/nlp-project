import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import warnings
warnings.filterwarnings('ignore')


class DocumentAnalyzer:
    def __init__(self):
        # download_nltk_data()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK fails
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
            
        self.word2vec_model = None
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.dictionary = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text with fallback tokenization"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Try NLTK tokenization first, fallback to simple split
            try:
                tokens = word_tokenize(text)
            except:
                # Fallback to simple whitespace tokenization
                tokens = text.split()
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    try:
                        lemmatized = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemmatized)
                    except:
                        # If lemmatization fails, use original token
                        processed_tokens.append(token)
            
            return processed_tokens
            
        except Exception as e:
            # Ultimate fallback - basic preprocessing
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            tokens = [word for word in text.split() if len(word) > 2]
            return tokens
    
    def train_word2vec(self, documents):
        """Train Word2Vec model"""
        try:
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            # Filter out empty documents
            processed_docs = [doc for doc in processed_docs if doc]
            
            if not processed_docs:
                raise ValueError("No valid documents to train on")
            
            self.word2vec_model = Word2Vec(
                sentences=processed_docs,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
            
            return self.word2vec_model
        except Exception as e:
            st.error(f"Error training Word2Vec model: {str(e)}")
            return None
    
    def get_document_vector(self, document):
        """Get document vector using Word2Vec"""
        tokens = self.preprocess_text(document)
        vectors = []
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)
    
    def calculate_similarity(self, documents):
        """Calculate document similarity using both Word2Vec and TF-IDF"""
        # Word2Vec similarity
        w2v_vectors = np.array([self.get_document_vector(doc) for doc in documents])
        w2v_similarity = cosine_similarity(w2v_vectors)
        
        # TF-IDF similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        tfidf_similarity = cosine_similarity(tfidf_matrix)
        
        return w2v_similarity, tfidf_similarity
    
    def perform_topic_modeling(self, documents, num_topics=5):
        """Perform topic modeling using LDA"""
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create dictionary and corpus
        self.dictionary = Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.7)
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        return self.lda_model
    
    def get_most_similar_words(self, word, top_n=10):
        """Get most similar words using Word2Vec"""
        try:
            if self.word2vec_model and hasattr(self.word2vec_model, 'wv') and word in self.word2vec_model.wv:
                return self.word2vec_model.wv.most_similar(word, topn=top_n)
        except Exception as e:
            print(f"Error finding similar words: {e}")
        return []

def main():
    st.set_page_config(
        page_title="NLP Document Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🔍 Advanced NLP Document Analysis Tool")
    st.markdown("**Analyze documents using Word2Vec, TF-IDF, and Topic Modeling**")
    
    # Initialize analyzer
    analyzer = DocumentAnalyzer()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        # Sample documents for demonstration
        sample_docs = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "Natural language processing helps computers understand and interpret human language in a valuable way.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "Data science combines statistics, programming, and domain knowledge to extract insights from data.",
            "Computer vision enables machines to interpret and understand visual information from the world.",
            "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence.",
            "Big data refers to extremely large datasets that require special tools and techniques to process.",
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "Cybersecurity protects digital systems, networks, and data from digital attacks and threats.",
            "Blockchain technology creates secure, decentralized digital ledgers for transactions."
        ]
        documents = sample_docs
        doc_names = [f"Document {i+1}" for i in range(len(sample_docs))]
    else:
        # File upload option
        st.sidebar.subheader("Upload Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Choose text files",
            type=['txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            documents = []
            doc_names = []
            for file in uploaded_files:
                try:
                    content = str(file.read(), "utf-8")
                    documents.append(content)
                    doc_names.append(file.name)
                except Exception as e:
                    st.sidebar.error(f"Error reading {file.name}: {str(e)}")
        else:
            st.warning("Please upload some text files or use sample data.")
            return
    
    # Check if we have valid documents
    if not documents or all(len(doc.strip()) == 0 for doc in documents):
        st.error("No valid documents found. Please check your input.")
        return
    
    # Configuration parameters
    num_topics = st.sidebar.slider("Number of Topics", 3, 10, 5)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.3)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔗 Document Similarity", 
        "🏷️ Topic Modeling", 
        "💭 Word Associations", 
        "☁️ Word Cloud"
    ])
    
    with tab1:
        st.header("Document Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(documents))
        
        with col2:
            avg_length = np.mean([len(doc.split()) for doc in documents])
            st.metric("Avg Words per Document", f"{avg_length:.0f}")
        
        with col3:
            total_words = sum([len(doc.split()) for doc in documents])
            st.metric("Total Words", total_words)
        
        # Document lengths visualization
        doc_lengths = [len(doc.split()) for doc in documents]
        
        # Create DataFrame for better handling
        length_df = pd.DataFrame({
            'Document': doc_names,
            'Word Count': doc_lengths
        })
        
        fig = px.bar(
            length_df,
            x='Document',
            y='Word Count',
            title="Document Lengths (Word Count)"
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show sample documents
        st.subheader("Document Preview")
        for i, (name, doc) in enumerate(zip(doc_names, documents)):
            with st.expander(f"{name} ({len(doc.split())} words)"):
                st.write(doc[:500] + "..." if len(doc) > 500 else doc)
    
    with tab2:
        st.header("Document Similarity Analysis")
        
        if st.button("Calculate Similarities", type="primary"):
            with st.spinner("Training Word2Vec model and calculating similarities..."):
                # Train Word2Vec
                model_result = analyzer.train_word2vec(documents)
                
                if model_result is None:
                    st.error("Failed to train Word2Vec model. Please check your documents.")
                    return
                
                # Calculate similarities
                try:
                    w2v_sim, tfidf_sim = analyzer.calculate_similarity(documents)
                except Exception as e:
                    st.error(f"Error calculating similarities: {str(e)}")
                    return
                
                # Create similarity heatmaps
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Word2Vec Similarity")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # Truncate long document names for display
                    display_names = [name[:15] + '...' if len(name) > 15 else name for name in doc_names]
                    sns.heatmap(w2v_sim, annot=True, fmt='.2f', 
                              xticklabels=display_names, yticklabels=display_names,
                              cmap='viridis', ax=ax)
                    plt.title("Word2Vec Document Similarity")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("TF-IDF Similarity")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(tfidf_sim, annot=True, fmt='.2f',
                              xticklabels=display_names, yticklabels=display_names,
                              cmap='plasma', ax=ax)
                    plt.title("TF-IDF Document Similarity")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Find most similar document pairs
                st.subheader("Most Similar Document Pairs")
                
                # Get upper triangle indices (avoid duplicates)
                triu_indices = np.triu_indices_from(w2v_sim, k=1)
                similarities = []
                
                for i, j in zip(triu_indices[0], triu_indices[1]):
                    similarities.append({
                        'Document 1': doc_names[i],
                        'Document 2': doc_names[j],
                        'Word2Vec Similarity': w2v_sim[i, j],
                        'TF-IDF Similarity': tfidf_sim[i, j]
                    })
                
                sim_df = pd.DataFrame(similarities)
                sim_df = sim_df.sort_values('Word2Vec Similarity', ascending=False)
                
                # Filter by threshold
                filtered_df = sim_df[sim_df['Word2Vec Similarity'] > similarity_threshold]
                
                if not filtered_df.empty:
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.info(f"No document pairs found with similarity > {similarity_threshold}")
    
    with tab3:
        st.header("Topic Modeling with LDA")
        
        if st.button("Perform Topic Analysis", type="primary"):
            with st.spinner("Training LDA model..."):
                lda_model = analyzer.perform_topic_modeling(documents, num_topics)
                
                # Display topics
                st.subheader("Discovered Topics")
                
                topics = []
                for idx, topic in lda_model.print_topics(num_words=8):
                    # Clean up topic string
                    words = re.findall(r'"([^"]*)"', topic)
                    topic_words = ', '.join(words[:6])
                    topics.append({
                        'Topic': f"Topic {idx + 1}",
                        'Keywords': topic_words
                    })
                
                topics_df = pd.DataFrame(topics)
                st.dataframe(topics_df, use_container_width=True)
                
                # Document-topic distribution
                st.subheader("Document Topic Distribution")
                
                doc_topics = []
                for i, doc in enumerate(documents):
                    processed_doc = analyzer.preprocess_text(doc)
                    bow = analyzer.dictionary.doc2bow(processed_doc)
                    topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
                    
                    topic_dist = [0] * num_topics
                    for topic_id, prob in topic_probs:
                        topic_dist[topic_id] = prob
                    
                    doc_topics.append(topic_dist)
                
                # Create heatmap for document-topic distribution
                topic_df = pd.DataFrame(
                    doc_topics,
                    columns=[f"Topic {i+1}" for i in range(num_topics)],
                    index=doc_names
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(topic_df, annot=True, fmt='.2f',
                          cmap='Blues', ax=ax)
                plt.title("Document-Topic Distribution")
                plt.xlabel("Topics")
                plt.ylabel("Documents")
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab4:
        st.header("Word Associations (Word2Vec)")
        
        # Train Word2Vec if not already done
        if analyzer.word2vec_model is None:
            if st.button("Train Word2Vec Model", type="primary"):
                with st.spinner("Training Word2Vec model..."):
                    analyzer.train_word2vec(documents)
                    st.success("Word2Vec model trained successfully!")
                    st.rerun()
            else:
                st.info("Click the button above to train the Word2Vec model first.")
                return
        
        # Get vocabulary
        vocab = []
        if analyzer.word2vec_model and hasattr(analyzer.word2vec_model, 'wv'):
            try:
                vocab = list(analyzer.word2vec_model.wv.key_to_index.keys())
            except:
                vocab = []
        
        if not vocab:
            st.warning("No vocabulary available. Please train the Word2Vec model first.")
            return
        
        # Word similarity search
        st.subheader("Find Similar Words")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_word = st.selectbox("Select a word:", sorted(vocab))
            num_similar = st.slider("Number of similar words:", 5, 20, 10)
        
        with col2:
            if selected_word:
                similar_words = analyzer.get_most_similar_words(selected_word, num_similar)
                
                if similar_words:
                    sim_data = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
                    
                    # Bar chart of similar words
                    fig = px.bar(
                        sim_data,
                        x='Similarity',
                        y='Word',
                        orientation='h',
                        title=f"Words most similar to '{selected_word}'",
                        color='Similarity',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No similar words found.")
        
        # Word arithmetic
        st.subheader("Word Arithmetic")
        st.markdown("Try word arithmetic like: king - man + woman = ?")
        
        if len(vocab) >= 10:  # Only show if we have enough vocabulary
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_words = st.multiselect("Positive words:", sorted(vocab)[:50])
            with col2:
                negative_words = st.multiselect("Negative words:", sorted(vocab)[:50])
            with col3:
                if st.button("Calculate") and analyzer.word2vec_model:
                    if positive_words:
                        try:
                            result = analyzer.word2vec_model.wv.most_similar(
                                positive=positive_words,
                                negative=negative_words,
                                topn=5
                            )
                            
                            result_df = pd.DataFrame(result, columns=['Word', 'Similarity'])
                            st.dataframe(result_df)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please select at least one positive word.")
        else:
            st.info("Word arithmetic requires a larger vocabulary. Train on more diverse documents.")
    
    with tab5:
        st.header("Word Clouds")
        
        # Generate word cloud for all documents
        all_text = ' '.join(documents)
        
        # Clean text for word cloud
        processed_text = ' '.join([' '.join(analyzer.preprocess_text(all_text))])
        
        if processed_text:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(processed_text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud - All Documents', fontsize=16)
            st.pyplot(fig)
            
            # Individual document word clouds
            st.subheader("Individual Document Word Clouds")
            
            selected_doc = st.selectbox("Select document:", doc_names)
            doc_index = doc_names.index(selected_doc)
            
            doc_text = ' '.join(analyzer.preprocess_text(documents[doc_index]))
            
            if doc_text:
                doc_wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='plasma',
                    max_words=50
                ).generate(doc_text)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(doc_wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Word Cloud - {selected_doc}', fontsize=16)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
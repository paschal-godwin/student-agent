import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

def generate_summary(vectorstore):

    if vectorstore:
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.3)

    cols = st.columns(4)
    with cols[0]:
        summary_topic = st.text_input("Enter a topic or chapter name you'd like summarized")
    with cols[1]:
        top_k_sum = st.number_input("Top-K context", min_value=3, max_value=20, value=8, step=1, key="summary_top_key")
    with cols[2]:
        target_words = st.number_input("Target length (words)", min_value=400, max_value=4000, value=1200, step=100)
    with cols[3]:
        reader_level = st.selectbox("Reader level", ["Beginner", "Intermediate", "Exam-focused"], index=1)

    include_aids = st.checkbox("Include study aids (key terms, tables, pitfalls, mini Q&A, mnemonics)", value=True)


    def _shield(text: str) -> str:

        # Avoid backtick confusion inside prompts
        return (text or "").replace("```", "\\`\\`\\`")

    def _chunk_digest_prompt(text: str) -> str:
        return f"""
    You are creating a structured digest from a textbook excerpt for later synthesis.

    Return Markdown only with these sections (no extra commentary):
    ### Section Candidates
    - (Proposed section title): 1–2 lines describing focus

    ### Salient Points
    - 6–12 bullets of the most important facts, mechanisms, definitions, steps

    ### Key Terms
    - term — 1-line definition

    ### Data/Formulae (if any)
    - concise list of equations, numeric ranges, rules

    ### Misconceptions (if any)
    - common confusion — correction

    EXCERPT:
    \"\"\"{_shield(text)}\"\"\"""".strip()

    def _final_summary_prompt(topic: str, combined_digests: str, target_words: int, reader_level: str, include_aids: bool) -> str:
        aids_block = """
    Include, after the detailed summary, these study aids:

    **Key Terms & Definitions** — thorough list.
    **Important Tables / Formulae / Ranges** — Markdown tables if useful.
    **Common Pitfalls & Misconceptions** — with corrections.
    **Mini Q&A (5–8)** — exam-style questions with brief answers.
    **Memory Hooks** — short mnemonics if natural.
    """.strip() if include_aids else ""

        return f"""
    You are a meticulous medical textbook summarizer. Produce a **comprehensive, structured Markdown** summary of the TOPIC below,
    synthesizing all information from the digests provided. Aim for ~{target_words} words (ok to exceed if needed for completeness).
    Audience: **{reader_level}**.

    Constraints & Style:
    - Be **faithful** to the digests (no external facts).
    - Organize with clear H2/H3 headings mirroring the topic's natural structure.
    - Use paragraphs (not just bullets). Add bullets or tables where appropriate.
    - Cover **all major sections** and how they connect; explain mechanisms and implications.
    - Prefer clarity and completeness over brevity.

    Output format (Markdown only):
    # {topic}
    ## Executive Overview
    (2–4 paragraphs that set the whole picture and why it matters)

    ## Detailed Outline
    (Numbered list of the main sections you will cover)

    ## Deep Dive
    (For each section: 2–6 paragraphs with subheadings, examples, edge cases, comparisons as relevant)

    ## Applications / Clinical or Practical Relevance (if applicable)

    {aids_block}

    ### Sources Used
    (List the source labels S1, S2... provided below)

    INPUT DIGESTS (S1..Sn, do not copy verbatim—synthesize):
    {combined_digests}
    """.strip()

    def _single_pass_prompt(topic: str, context: str, target_words: int, reader_level: str, include_aids: bool) -> str:
        aids_block = """
    Also include, after the deep dive:
    - Key Terms & Definitions
    - Important Tables / Formulae / Ranges
    - Common Pitfalls & Misconceptions
    - Mini Q&A (5–8) with brief answers
    - Memory Hooks (mnemonics, if natural)
    """.strip() if include_aids else ""

        return f"""
    Summarize the TOPIC comprehensively from the CONTEXT. Target ~{target_words} words (ok to exceed for completeness).
    Audience: **{reader_level}**.

    Write **structured Markdown** with:
    # {topic}
    ## Executive Overview
    ## Detailed Outline
    ## Deep Dive (sectioned, multi-paragraph)
    ## Applications / Clinical or Practical Relevance (if any)
    {aids_block}

    Rules:
    - Ground strictly in the CONTEXT (no outside facts).
    - Be detailed and explanatory, not terse.
    - Use headings, paragraphs, and well-placed bullets/tables.

    TOPIC:
    \"\"\"{_shield(topic)}\"\"\"\n
    CONTEXT:
    \"\"\"{_shield(context)}\"\"\"""".strip()

    def _collect_labels(docs):
        labels = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}

            # 1. Try to get the file name
            name = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("title")
                or f"Document {i}"
            )

            # 2. Add page number if available
            page = meta.get("page")
            if page is not None:
                name = f"{name} (Page {page})"

            # 3. Add a text snippet for clarity
            snippet = d.page_content[:60].replace("\n", " ")
            if snippet:
                name = f"{name}: \"{snippet}...\""

            labels.append((f"S{i}", name))
        return labels


    def _map_reduce_summarize(topic: str, docs, target_words: int, reader_level: str, include_aids: bool):
        # Map step: per-chunk digests
        digests = []
        for d in docs:
            excerpt = d.page_content
            prompt = _chunk_digest_prompt(excerpt)
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            digests.append(text.strip())

        # Label digests as S1..Sn
        labeled = [f"---\n**[S{i}] Digest**\n\n{dig}" for i, dig in enumerate(digests, start=1)]
        combined = "\n\n".join(labeled)

        # Reduce step: synthesize final comprehensive summary
        final_prompt = _final_summary_prompt(topic, combined, target_words, reader_level, include_aids)
        final_resp = llm.invoke(final_prompt)
        return final_resp.content if hasattr(final_resp, "content") else str(final_resp)

    def _single_pass_summarize(topic: str, docs, target_words: int, reader_level: str, include_aids: bool):
        context = "\n\n".join([d.page_content for d in docs])
        prompt = _single_pass_prompt(topic, context, target_words, reader_level, include_aids)
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    def _generate_summary(topic: str, top_k: int, target_words: int, reader_level: str, include_aids: bool):
        relevant_docs = retriever.get_relevant_documents(topic)
        docs = relevant_docs[:top_k]

        total_chars = sum(len(d.page_content) for d in docs)
        # Heuristic: large => map-reduce; small => single-pass
        if total_chars > 8000 or len(docs) > 6:
            return _map_reduce_summarize(topic, docs, target_words, reader_level, include_aids), docs
        else:
            return _single_pass_summarize(topic, docs, target_words, reader_level, include_aids), docs

    # --- UI / Execution ---
    if summary_topic:
        if vectorstore:
            with st.spinner("Retrieving and summarizing from textbook context..."):
                try:
                    summary_md, used_docs = _generate_summary(
                        summary_topic, int(top_k_sum), int(target_words), reader_level, include_aids
                    )

                    # Basic sanity check
                    if not summary_md or len(summary_md.strip()) < 200:
                        st.warning("Summary seems too short. Consider increasing Top-K or Target length.")
                    
                    # Display
                    st.markdown("### 📘 Summary")
                    st.markdown(summary_md)

                    # Sources
                    labels = _collect_labels(used_docs)
                    with st.expander("Sources used"):
                        for tag, name in labels:
                            st.markdown(f"- **{tag}** — {name}")

                    # Download
                    file_name = f"summary_{summary_topic.replace(' ', '_')}.md"
                    st.download_button(
                        "⬇️ Download Markdown",
                        data=summary_md.encode("utf-8"),
                        file_name=file_name,
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")
        else:
            st.error("Please upload or load study materials first.")
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


def generate_flashcard(vectorstore):

    if vectorstore:
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.3)

    # optional: let user tune number of cards and top_k retrieval
    cols = st.columns(3)
    with cols[0]:
        flashcard_topic = st.text_input("Enter a topic you'd like to generate flashcards for")
    with cols[1]:
        num_cards = st.number_input("Cards", min_value=5, max_value=50, value=10, step=1)
    with cols[2]:
        top_k = st.number_input("Top-K context", min_value=3, max_value=20, value=8, step=1, key= "flashcard_top_key")

    # init UI state
    if "fcards" not in st.session_state:
        st.session_state.fcards = []          # list of dicts: {question, answer, source_span}
    if "fc_idx" not in st.session_state:
        st.session_state.fc_idx = 0
    if "fc_show_ans" not in st.session_state:
        st.session_state.fc_show_ans = False
    if "fc_ratings" not in st.session_state:
        st.session_state.fc_ratings = {}      # idx -> again|hard|good|easy

    def _next_card():
        if st.session_state.fcards:
            st.session_state.fc_idx = min(st.session_state.fc_idx + 1, len(st.session_state.fcards) - 1)
            st.session_state.fc_show_ans = False

    def _prev_card():
        if st.session_state.fcards:
            st.session_state.fc_idx = max(st.session_state.fc_idx - 1, 0)
            st.session_state.fc_show_ans = False

    def _rate(r):
        st.session_state.fc_ratings[st.session_state.fc_idx] = r
        _next_card()

    def _build_prompt(topic: str, context: str, n: int) -> str:
        return f"""
    You are a study assistant generating **Anki-style** flashcards strictly from the provided textbook context.

    GOAL:
    - Create {n} high-quality flashcards that test understanding (not trivial recall).
    - Each card must be grounded in the CONTEXT; do not invent facts.

    FORMAT:
    Return ONLY a JSON list (no prose) where each item is an object:
    {{
    "question": "...",   // one atomic, self-contained question
    "answer": "...",     // short, factual answer (bullet points allowed)
    "source_span": "..." // exact phrase/sentence(s) from CONTEXT that justify the answer
    }}

    GUIDELINES:
    - Prefer conceptual questions (why/how/compare) when possible, mixed with precise definitions.
    - Make questions self-contained: include necessary terms (no "this/that").
    - Keep answers concise (<= 4 bullet points if needed).
    - Absolutely no text outside the JSON.

    TOPIC:
    \"\"\"{topic}\"\"\"

    CONTEXT (textbook excerpts):
    \"\"\"{context}\"\"\"
    """

    def _parse_cards(raw: str):
        import json, re
        try:
            # sometimes models wrap JSON in code fences
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(raw)
            out = []
            if isinstance(data, list):
                for c in data:
                    q = (c.get("question") or "").strip()
                    a = (c.get("answer") or "").strip()
                    s = (c.get("source_span") or "").strip()
                    if q and a:
                        out.append({"question": q, "answer": a, "source_span": s})
            return out
        except Exception:
            return []

    def _to_anki_tsv(cards):
        # Question<TAB>Answer format
        import io
        buf = io.StringIO()
        for c in cards:
            q = c["question"].replace("\n", " ").strip()
            a = c["answer"].strip()
            buf.write(f"{q}\t{a}\n")
        return buf.getvalue()

    def _generate_cards(topic: str, k: int, n: int):
        # retrieve textbook chunks for THIS topic
        relevant_docs = retriever.get_relevant_documents(topic)
        # limit to top_k
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:k]])

        prompt = _build_prompt(topic, context, n)
        # Use your existing LLM wrapper
        result = llm.invoke(prompt)
        raw = result.content if hasattr(result, "content") else str(result)
        cards = _parse_cards(raw)

        # Fallback: if model ignored JSON instruction, try your old TSV approach
        if not cards:
            fallback_prompt = f"""
    You are a study assistant generating Anki-style flashcards.

    - Based ONLY on the CONTEXT below and the TOPIC, create {n} flashcards.
    - Format: "Question<TAB>Answer" per line.
    - No numbering, no extra text.

    TOPIC: {topic}

    CONTEXT:
    {context}

    OUTPUT:
    """
            fb = llm.invoke(fallback_prompt)
            text = fb.content.strip()
            # parse tsv lines
            lines = [ln for ln in text.split("\n") if "\t" in ln]
            for ln in lines:
                q, a = ln.split("\t", 1)
                cards.append({"question": q.strip(), "answer": a.strip(), "source_span": ""})

        return cards

    # UI
    if flashcard_topic:
        if vectorstore:
            if st.button("Generate flashcards"):
                with st.spinner("Generating flashcards from textbook context..."):
                    cards = _generate_cards(flashcard_topic, int(top_k), int(num_cards))
                    if not cards:
                        st.error("Could not generate flashcards. Try a different topic or increase Top-K.")
                    else:
                        st.session_state.fcards = cards
                        st.session_state.fc_idx = 0
                        st.session_state.fc_show_ans = False
                        st.session_state.fc_ratings = {}

            # Render deck
            cards = st.session_state.fcards
            if cards:
                idx = st.session_state.fc_idx
                card = cards[idx]
                total = len(cards)

                st.caption(f"Card {idx+1} / {total}")
                st.markdown("### Question")
                st.write(card["question"])

                # reveal button
                if not st.session_state.fc_show_ans:
                    if st.button("Show answer"):
                        st.session_state.fc_show_ans = True
                else:
                    st.markdown("### Answer")
                    st.write(card["answer"])
                    with st.expander("Source span (from textbook context)"):
                        st.write(card.get("source_span") or "_No span provided_")

                    st.markdown("**Rate this card**")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if st.button("🔁 Again"):
                            _rate("again")
                    with c2:
                        if st.button("⚠️ Hard"):
                            _rate("hard")
                    with c3:
                        if st.button("✅ Good"):
                            _rate("good")
                    with c4:
                        if st.button("🟢 Easy"):
                            _rate("easy")

                # nav + progress
                n1, n2, bar = st.columns([1,1,6])
                with n1:
                    st.button("⟵ Prev", on_click=_prev_card, disabled=(idx == 0))
                with n2:
                    st.button("Next ⟶", on_click=_next_card, disabled=(idx == total-1))
                with bar:
                    st.progress((idx + 1) / total)

                st.divider()

                # Export buttons
                tsv_data = _to_anki_tsv(cards)
                st.download_button(
                    "⬇️ Export Anki TSV",
                    data=tsv_data.encode("utf-8"),
                    file_name=f"flashcards_{flashcard_topic.replace(' ','_')}.tsv",
                    mime="text/tab-separated-values"
                )

                with st.expander("Session ratings (debug)"):
                    st.json(st.session_state.fc_ratings)
        else:
            st.error("Please upload or load a study document first.")

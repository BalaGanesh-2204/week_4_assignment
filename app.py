"""
Streamlit UI for the Bala Support e-commerce agent.

- Chat with the agent (structured answers rendered per turn).
- Sidebar "For Balaganesh": open escalations + reminders with resolve
  buttons, customer binding, admin actions.
- Trace viewer: browse every logged turn JSON.
"""

import streamlit as st

import agent_logger
import harness
import ingest
import memory_store
import reminders

from config import STORE_NAME


st.set_page_config(
    page_title=f"{STORE_NAME} Support Agent",
    page_icon="🛒",
    layout="centered",
)


# ---------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------

if "session_id" not in st.session_state:

    st.session_state.session_id = agent_logger.new_session_id()

    try:
        memory_store.register_session(st.session_state.session_id)
    except Exception:
        pass

if "messages" not in st.session_state:
    st.session_state.messages = []

if "customer_email" not in st.session_state:
    st.session_state.customer_email = ""


SESSION_ID = st.session_state.session_id


# ---------------------------------------------------------
# SIDEBAR - FOR BALAGANESH
# ---------------------------------------------------------

with st.sidebar:

    st.title(f"🛒 {STORE_NAME} Support")
    st.caption(f"Session `{SESSION_ID}`")

    st.divider()

    # ----- Reminder / escalation board -----

    counts = reminders.counts()
    st.subheader("📌 For Balaganesh")

    if counts["total"] == 0:
        st.info("No open escalations or reminders.")
    else:
        st.markdown(
            f"**{counts['escalations']}** escalation(s) · "
            f"**{counts['reminders']}** reminder(s) · "
            f"**{counts['critical']}** critical"
        )

        for entry in reminders.open_entries():

            icon = {
                "critical": "🚨",
                "high": "❗",
                "medium": "🔔",
                "low": "💬",
            }.get(entry["urgency"], "🔔")

            label = (
                f"{icon} [{entry['type'].upper()}] "
                f"{entry['message'][:80]}"
            )

            with st.expander(label):
                st.write(entry["message"])
                st.caption(
                    f"id {entry['id']} · urgency {entry['urgency']} · "
                    f"created {entry['created_at'][:19]}"
                )
                if st.button(
                    "Mark resolved",
                    key=f"resolve_{entry['id']}",
                ):
                    reminders.resolve_entry(entry["id"])
                    st.rerun()

    st.divider()

    # ----- Customer binding -----

    st.subheader("👤 Customer binding")

    email_input = st.text_input(
        "Customer email",
        value=st.session_state.customer_email,
        placeholder="anita.sharma@example.com",
    )

    if email_input != st.session_state.customer_email:
        st.session_state.customer_email = email_input.strip()
        st.rerun()

    if st.session_state.customer_email:

        try:
            facts = memory_store.recall_facts(
                st.session_state.customer_email.lower()
            )
        except Exception:
            facts = []

        st.caption(
            f"Bound to `{st.session_state.customer_email}` · "
            f"{len(facts)} remembered fact(s)"
        )
    else:
        st.caption("Guest mode - no long-term memory binding.")

    st.divider()

    # ----- Admin actions -----

    with st.expander("⚙️ Admin"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Re-ingest docs"):
                with st.spinner("Ingesting..."):
                    try:
                        stats = ingest.run_ingest()
                        st.success(
                            f"{stats['upserted']} vectors upserted "
                            f"({stats['chunks']} chunks)."
                        )
                    except Exception as exc:
                        st.error(f"Ingest failed: {exc}")

        with col2:
            if st.button("Pinecone stats"):
                try:
                    import vector_store

                    st.json(vector_store.get_index_stats())
                except Exception as exc:
                    st.error(f"Stats failed: {exc}")

        if st.button("Reset this chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button(
            "Forget bound customer's memory",
            disabled=not st.session_state.customer_email,
        ):
            try:
                removed = memory_store.forget_customer(
                    st.session_state.customer_email.lower()
                )
                st.success(f"Deleted {removed} fact(s).")
            except Exception as exc:
                st.error(f"Failed: {exc}")

    st.divider()

    # ----- Trace viewer -----

    st.subheader("🕵️ Trace logs")

    sessions = agent_logger.get_sessions()

    if not sessions:
        st.caption("No logged sessions yet.")
    else:

        chosen_session = st.selectbox(
            "Session",
            sessions,
            index=(
                sessions.index(SESSION_ID)
                if SESSION_ID in sessions
                else 0
            ),
        )

        turn_files = agent_logger.get_turn_logs(chosen_session)

        if turn_files:

            labels = [f.name for f in turn_files]

            chosen_turn = st.selectbox("Turn", labels)

            log = agent_logger.read_log(
                next(
                    f for f in turn_files if f.name == chosen_turn
                )
            )

            with st.expander("Raw turn log", expanded=False):
                st.json(log)

        else:
            st.caption("No turns logged for this session.")


# ---------------------------------------------------------
# CHAT UI
# ---------------------------------------------------------

st.title("💬 Customer Support")
st.caption(
    "Ask about orders, shipping, returns, refunds, promos or "
    "products. Escalations reach the human support lead."
)


def _render_meta(meta: dict):
    """
    Expander under an assistant bubble with structured details.
    """

    lines = [
        f"intent: `{meta.get('intent')}`",
        f"confidence: `{meta.get('confidence')}`",
        f"steps used: `{meta.get('steps_used')}`",
    ]

    if meta.get("tools_used"):
        lines.append("tools: " + ", ".join(f"`{t}`" for t in meta["tools_used"]))

    st.markdown(" · ".join(lines))

    if meta.get("citations"):
        st.markdown("**Sources**")
        for citation in meta["citations"]:
            heading = citation.get("heading") or ""
            st.markdown(
                f"- 📄 `{citation.get('source')}`"
                + (f" — {heading}" if heading else "")
            )

    flags = []
    if meta.get("escalated"):
        flags.append("🚨 escalated to Balaganesh")
    if meta.get("needs_human"):
        flags.append("🙋 needs human follow-up")

    if flags:
        st.warning(" · ".join(flags))

    if meta.get("log_path"):
        st.caption(f"turn log: `{meta['log_path']}`")


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("meta"):
            with st.expander("Turn details", expanded=False):
                _render_meta(message["meta"])


user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                outcome = harness.run_turn(
                    session_id=SESSION_ID,
                    user_input=user_input,
                    customer_email=(
                        st.session_state.customer_email.strip() or None
                    ),
                )

                final_answer = outcome["final_answer"]

                meta = final_answer.model_dump()
                meta["steps_used"] = outcome["steps_used"]
                meta["log_path"] = outcome["log_path"]

            except Exception as exc:

                meta = None

                st.error(
                    "The support service hit an error. Please retry.\n\n"
                    f"`{exc}`"
                )

    if meta is not None:

        content = meta["answer"]

        st.session_state.messages.append(
            {"role": "assistant", "content": content, "meta": meta}
        )

        st.rerun()

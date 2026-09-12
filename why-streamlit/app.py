import json
import os
import uuid
from datetime import datetime

import streamlit as st

STATE_FILE = "why_state.json"


def now():
    return datetime.now().isoformat(timespec="seconds")


def new_state():
    return {
        "version": 1,
        "updated_at": now(),
        "problem": None,
        "goal": None,
        "representation": None,
        "discoveries": [],
        "questions": [],
        "history": [],
    }


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return new_state()


# ---------------------------
# Initialisation session
# ---------------------------
if "state" not in st.session_state:
    st.session_state.state = load_state()

if "chat" not in st.session_state:
    st.session_state.chat = []
    if not st.session_state.state["history"]:
        st.session_state.chat.append(("WHY", "Je suis prêt. Donne-moi le problème."))

state = st.session_state.state


def save():
    state["updated_at"] = now()
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        # Le système de fichiers peut être en lecture seule (ex. Streamlit Cloud) :
        # l'état vit quand même dans la session.
        pass


def add_history(kind, text, **extra):
    state["history"].append(
        {"id": uuid.uuid4().hex[:8], "time": now(), "kind": kind, "text": text, **extra}
    )


def pending():
    return [q for q in state["questions"] if q["status"] == "pending"]


def ask(text, reason="", kind="normal"):
    for q in pending():
        if q["text"].casefold() == text.casefold():
            return q["id"]
    q = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "reason": reason,
        "kind": kind,
        "status": "pending",
        "answer": None,
    }
    state["questions"].append(q)
    add_history("question_asked", text, question_id=q["id"])
    return q["id"]


def answer(q, text):
    q["status"] = "answered"
    q["answer"] = text
    q["answered_at"] = now()
    add_history("question_answered", text, question_id=q["id"])

    # Si cette question servait à obtenir le but de l'utilisateur,
    # sa réponse devient automatiquement le but.
    if q.get("kind") == "goal_clarification" or q.get("reason") == "But utilisateur absent.":
        state["goal"] = text
        add_history("goal", text, source="answer_to_clarification", question_id=q["id"])


def show(who, text):
    st.session_state.chat.append((who, text))


def execute():
    ps = pending()
    if ps:
        show("WHY", f"J'attends encore ta réponse à :\n« {ps[0]['text']} »")
        return
    if not state["problem"]:
        show("WHY", "Je n'ai pas encore de problème à traiter.")
        return
    if not state["goal"]:
        ask(
            "Qu'est-ce que tu veux comprendre, décider ou obtenir à partir de ce problème ?",
            "But utilisateur absent.",
            kind="goal_clarification",
        )
        show("WHY", state["questions"][-1]["text"])
        save()
        return
    if not state["representation"]:
        state["representation"] = (
            "Représentation initiale construite à partir du problème et du but utilisateur."
        )
        add_history("representation_update", state["representation"])
        show("WHY", "J'ai mis à jour ma représentation du problème.")
        save()
        return
    show("WHY", "Aucune nouvelle action n'est justifiée par l'état actuel.")
    save()


def process(text):
    text = text.strip()
    if not text:
        return
    add_history("user_message", text)

    ps = pending()
    if ps:
        q = ps[0]
        answer(q, text)
        if state.get("goal") == text and (
            q.get("kind") == "goal_clarification"
            or q.get("reason") == "But utilisateur absent."
        ):
            show(
                "WHY",
                "J'ai compris : « " + text + " » est ton but pour ce problème. "
                "Je l'enregistre comme tel.",
            )
        else:
            show("WHY", "Réponse intégrée à mon état. Je ne reposerai pas cette question.")
        save()
        return

    if not state["problem"]:
        state["problem"] = text
        add_history("problem", text)
        ask(
            "Qu'est-ce que tu veux comprendre, décider ou obtenir à partir de ce problème ?",
            "But utilisateur absent.",
            kind="goal_clarification",
        )
        show("WHY", state["questions"][-1]["text"])
        save()
        return

    low = text.lower()
    if low.startswith(("je veux ", "mon but ", "je cherche à ", "en fait je veux ")):
        state["goal"] = text
        add_history("goal", text)
        show("WHY", "But utilisateur enregistré.")
        save()
        return

    state["discoveries"].append({"id": uuid.uuid4().hex[:8], "text": text, "time": now()})
    show("WHY", "Information ajoutée à mon état.")
    save()


def send(text):
    text = text.strip()
    if not text:
        return
    show("TOI", text)
    low = text.lower()
    if low == "/exec":
        execute()
    elif low == "/etat":
        ps = pending()
        show(
            "WHY",
            f"Problème : {state['problem'] or '—'}\n"
            f"But : {state['goal'] or '—'}\n"
            f"Représentation : {state['representation'] or '—'}\n"
            f"Découvertes : {len(state['discoveries'])}\n"
            f"Questions en attente : {len(ps)}",
        )
    elif low == "/reset":
        state.clear()
        state.update(new_state())
        save()
        show("WHY", "État réinitialisé.")
    else:
        process(text)


# ---------------------------
# Interface Streamlit
# ---------------------------
st.set_page_config(page_title="WHY", page_icon="🧠", layout="centered")
st.title("WHY")

# Affichage de l'historique de conversation
for who, text in st.session_state.chat:
    with st.chat_message("user" if who == "TOI" else "assistant"):
        st.markdown(f"**{who}**\n\n{text}")

# Zone de saisie
prompt = st.chat_input("Ton message (commandes : /exec, /etat, /reset)")
if prompt:
    send(prompt)
    st.rerun()

# Barre latérale : état courant
with st.sidebar:
    st.header("État")
    st.write(f"**Problème :** {state['problem'] or '—'}")
    st.write(f"**But :** {state['goal'] or '—'}")
    st.write(f"**Représentation :** {state['representation'] or '—'}")
    st.write(f"**Découvertes :** {len(state['discoveries'])}")
    st.write(f"**Questions en attente :** {len(pending())}")
    st.caption(f"Dernière mise à jour : {state['updated_at']}")

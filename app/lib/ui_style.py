import streamlit as st

def hero(title: str, subtitle: str):
    st.markdown(f"# {title}")
    st.caption(subtitle)

def section(title: str, desc: str | None = None, icon: str | None = None):
    t = f"## {icon+' ' if icon else ''}{title}"
    st.markdown(t)
    if desc:
        st.caption(desc)

def pill(text: str):
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;"
        f"background:#f3f4f6;border:1px solid #e5e7eb;font-size:12px;'>{text}</span>",
        unsafe_allow_html=True,
    )

def kpi(value: str, label: str, help_text: str | None = None):
    st.metric(label=label, value=value, help=help_text)

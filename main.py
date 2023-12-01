import json
import os

from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent, px
import streamlit as st
import predictionguard as pg

#---------------------#
# Streamlit things    #
#---------------------#

#st.set_page_config(layout="wide")

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model_dict = {
    "Nous-Hermes-Llama2-13B (Text Generation)": "Nous-Hermes-Llama2-13B", 
    "Neural-Chat-7B (Chat)": "Neural-Chat-7B", 
    "WizardCoder-15B (Code Generation)": "WizardCoder", 
    "Yi-34B (Text Generation)": "Yi-34B",
    "Zephyr-7B-Beta (Chat)": "Zephyr-7B"
    }

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


#---------------------------------#
# Footer functionality            #
#---------------------------------#

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="60px",
        opacity=0.6
    )

    style_hr = styles(
    )
    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)
    st.markdown(style, unsafe_allow_html=True)
    
    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)
def footer():
    myargs = [
        "<b>Made with</b>: Prediction Guard ",
        link("https://www.predictionguard.com/", image('https://www.sweasy26.com/emoji-dictionary/images/white-chess-rook.png',
        	width=px(18), height=px(18), margin= "0em")),
        br(),
        "(a member of Intel LiftOff)",
        br(),
    ]
    layout(*myargs)


if __name__ == "__main__":
    if check_password():
        st.title("♜ Prediction Guard Playground")
        completions_tab, factuality_tab, toxicity_tab = st.tabs(["Completions", "Factuality", "Toxicity"])
        
        with completions_tab:
            prompt = st.text_area("Enter an LLM prompt", height=200, key="prompt")
            with st.expander("Model Control and configuration"):
                model = st.selectbox("Model", [
                    "Neural-Chat-7B (Chat)",
                    "Nous-Hermes-Llama2-13B (Text Generation)", 
                    "WizardCoder-15B (Code Generation)", 
                    "Yi-34B (Text Generation)",
                    "Zephy-7B-Beta (Chat)"
                    ])
                consistency = st.checkbox("Consistency", key="consistency", value=False)
                factuality = st.checkbox("Factuality", key="factuality_comp", value=False)
                toxicity = st.checkbox("Toxicity", key="toxicity_comp", value=False)

                # parse output_types into json dict
                output = {}
                output["consistency"] = consistency
                output["factuality"] = factuality
                output["toxicity"] = toxicity

            with st.expander("Text Completion Parameters"):
                temperature = st.slider("Temperature", 0.0, 1.0, 0.75)
                top_p = st.slider("Top-p", 0.0, 1.0, 0.9)
                max_tokens = st.slider("Max Tokens", 0, 1024, 100)

            if st.button("Generate", key="comp_spinner"):
                with st.spinner("Generating..."):
                    print(output)
                    result = pg.Completion.create(
                        model=model_dict[model],
                        prompt=prompt,
                        output=output,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    if 'error' in result['choices'][0]['status']:
                        st.warning(result['choices'][0]['status'])
                    else:
                        st.success(result['choices'][0]['text'])
        
        with factuality_tab:
            text = st.text_area("Enter Text to Check for Factuality", height=100, key="fact_text")
            ref = st.text_area("Enter a Reference to Check the Text Against", height=100, key="fact_ref")

            if st.button("Generate", key="fact_button"):
                with st.spinner("Generating..."):
                    result = pg.Factuality.check(
                        reference=ref,
                        text=text
                    )
                    if 'error' in result['checks'][0]['status']:
                        st.warning(result['checks'][0]['status'])
                    else:
                        st.success("Score: " + str(result['checks'][0]['score']))
                        st.progress(result['checks'][0]['score'])

        with toxicity_tab:
            text = st.text_area("Enter Text to Check for Toxicity", height=200, key="tox_text")

            if st.button("Generate", key="tox_button"):
                with st.spinner("Generating..."):
                    result = pg.Toxicity.check(
                        text=text
                    )
                    if 'error' in result['checks'][0]['status']:
                        st.warning(result['checks'][0]['status'])
                    else:
                        st.success("Score: " + str(result['checks'][0]['score']))
                        st.progress(result['checks'][0]['score'])

        footer()
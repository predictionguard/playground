import json
import os

from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent, px
import streamlit as st

#os.environ["PREDICTIONGUARD_URL"] = "https://staging.predictionguard.com"

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
    "Nous-Hermes-2-SOLAR-10.7B (Text Generation)": "Nous-Hermes-2-SOLAR-10.7B", 
    "Neural-Chat-7B (Chat)": "Neural-Chat-7B",
    "deepseek-coder-6.7b-instruct (Code/SQL Generation, Tech Assistant)": "deepseek-coder-6.7b-instruct",
    "Yi-34B-Chat (English + Mandarin Chat)": "Yi-34B-Chat",
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
            with st.expander("Model validation and configuration"):
                st.markdown("#### Model:")
                model = st.selectbox("Model", model_dict.keys())
                st.markdown("#### Input Filters:")
                pii = st.checkbox("PII", key="pii", value=False)
                prompt_injection = st.checkbox("Prompt Injection", key="prompt_injection", value=False)
                st.markdown("#### Output Checks:")
                consistency = st.checkbox("Consistency", key="consistency", value=False)
                #factuality = st.checkbox("Factuality", key="factuality_comp", value=False)
                toxicity = st.checkbox("Toxicity", key="toxicity_comp", value=False)

                # parse output_types into json dict
                output = {}
                output["consistency"] = consistency
                #output["factuality"] = factuality
                output["toxicity"] = toxicity

            with st.expander("Text Completion Parameters"):
                temperature = st.slider("Temperature", 0.0, 1.0, 0.75)
                #top_p = st.slider("Top-p", 0.0, 1.0, 0.9)
                max_tokens = st.slider("Max Tokens", 0, 1024, 100)
            if len(prompt.split(" ")) > 1500:
                st.warning("Max prompt length exceeded for playground environment. Please try a shorter prompt.")
            else:
                if st.button("Generate", key="comp_spinner"):
                    with st.spinner("Generating..."):

                        completion = ""

                        # Check for PII
                        if pii:
                            with st.spinner("Checking for PII..."):
                                pii_response = pg.PII.check(
                                    prompt=prompt,
                                    replace=False,
                                    replace_method="fake"
                                )
                                if "[" in pii_response['checks'][0]['pii_types_and_positions']:
                                    pii_result = True
                                    st.warning("Warning! PII detected. Please avoid using personal information.")
                                else:
                                    pii_result = False
                        else:
                            pii_result = False

                        # Check for injection
                        if prompt_injection:
                            with st.spinner("Checking for security vulnerabilities..."):
                                injection_response = pg.Injection.check(
                                    prompt=prompt,
                                    detect=True
                                )
                                if injection_response['checks'][0]['probability'] > 0.5:
                                    injection_result = True
                                    st.warning("Warning! Security vulnerabilities detected. Please avoid using malicious prompts.")
                                else:
                                    injection_result = False
                        else:
                            injection_result = False

                        if not pii_result and not injection_result:
                            result = pg.Completion.create(
                                model=model_dict[model],
                                prompt=prompt,
                                output=output,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            if 'error' in result['choices'][0]['status']:
                                st.warning(result['choices'][0]['status'])
                            else:
                                st.success(result['choices'][0]['text'])
        
        with factuality_tab:
            text = st.text_area("Draft text (to check against a reference)", height=100, key="fact_text")
            ref = st.text_area("Reference text (to validate the draft)", height=100, key="fact_ref")

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
            text = st.text_area("Text to check for toxicity", height=200, key="tox_text")

            if st.button("Generate", key="tox_button"):
                with st.spinner("Generating..."):
                    result = pg.Toxicity.check(
                        text=text
                    )
                    if 'error' in result['checks'][0]['status']:
                        st.warning(result['checks'][0]['status'])
                    else:
                        st.success("Score: " + str(round(result['checks'][0]['score'], 4)))
                        st.progress(result['checks'][0]['score'])

        footer()
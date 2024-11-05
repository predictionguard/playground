# Example demonstrating all of the Prediction Guard OPEA components

This example spins up a Streamlit UI that lets you try all of the Prediction Guard based OPEA components. 

## Get a Prediction Guard API key

For OPEA community members, Prediction Guard is offering FREE evaluation API keys to try out OPEA components and the Prediction Guard API. Join our [Discord server](https://discord.com/invite/TFHgnhAFKd) and mention that you are experimenting with OPEA to get a key. One of our team members will follow up.

## 🏗️ Build Docker Images

First of all, you need to build the respective Docker Images (or already have them built in your registry).

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
```

Then you can execute any of the following to build LLM, LVM, embedding, and guardrails service images:

```bash
docker build -t opea/llm-predictionguard:latest -f comps/llms/text-generation/predictionguard/docker/Dockerfile .
docker build -t opea/lvm-predictionguard:latest -f comps/lvms/predictionguard/Dockerfile .
docker build -t opea/embedding-predictionguard:latest -f comps/embeddings/predictionguard/docker/Dockerfile .
docker build -t opea/factuality-predictionguard:latest -f comps/guardrails/factuality/predictionguard/docker/Dockerfile .
docker build -t opea/pii-predictionguard:latest -f comps/guardrails/pii_detection/predictionguard/docker/Dockerfile .
docker build -t opea/injection-predictionguard:latest -f comps/guardrails/prompt_injection/predictionguard/docker/Dockerfile .
docker build -t opea/toxicity-predictionguard:latest -f comps/guardrails/toxicity_harm/predictionguard/docker/Dockerfile .
```

## 🏃‍♀️ Run Docker Images

1. Define environment variables:

    ```bash
    export PREDICTIONGUARD_API_KEY=<your api key>
    ```

2. Deploy the OPEA components:

    ```bash
    docker compose up
    ```

## Run the Web App playground locally

The streamlit web app can be run locally as shown below or via the Streamlit Cloud. To run the web app locally:

```bash
pip install requirements.txt
streamlit run ui.py
```
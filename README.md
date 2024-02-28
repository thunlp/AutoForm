## Getting Started

### Installation

Clone the repository and install the dependencies.

```bash
git clone git@github.com:thunlp/AutoForm.git
cd AutoForm
# Python>=3.10
pip install -r requirements.txt
```

### Environment Variables

Set OpenAI key and Google API key.

```bash
# Export your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Export your Google API key
export GOOGLE_API_KEY="your_api_key_here"
```

### Run the scripts

Here is an example of how to run the scripts. This script implement **Single-LLM Reasoning** using **Gemini Pro** with **AutoForm** on **AQuA** dataset.

```bash
python scripts/aqua/autoform-gemini.py
```
# Emotion-Aware Playlist & Quote Recommender

Sometimes it helps to put a feeling into words. This project takes a short
piece of text, estimates the emotion behind it, and suggests a few songs and
a quote to go with that mood. If you would rather shift the mood, choose the
“Lift my mood” option instead.

The project is built as a hands-on NLP notebook: it explores the data, compares
a traditional text classifier with a fine-tuned transformer, and finishes with
a small interactive demo.

## What it does

- Predicts probabilities for six emotions: sadness, joy, love, anger, fear, and
  surprise.
- Recommends songs and a quote that match the predicted emotion, or picks a
  target emotion for the “Lift my mood” mode.
- Includes text normalization, exploratory analysis, model evaluation, and a
  Gradio interface in one notebook.

The song and quote collections are curated examples in the notebook, not live
catalogue searches.

## The notebook

Open [`nlp-project.ipynb`](./nlp-project.ipynb) in Kaggle or another Jupyter
environment. Kaggle is the easiest way to get started, since the notebook was
prepared with its hosted GPU environment in mind.

For Kaggle, create a new notebook, upload this file, then turn on Internet and
select a GPU accelerator (a T4 or P100 is suggested). Run the cells from top to
bottom. The notebook downloads the emotion dataset and pretrained model when
they are first needed, so Internet access must remain enabled.

The notebook installs a few packages near the beginning. If you are using your
own environment instead, install the project dependencies before running:

```bash
pip install -r requirements.txt
```

PyTorch's best installation command can depend on your operating system and
whether you want GPU support; use the [official PyTorch install selector](https://pytorch.org/get-started/locally/)
if installing PyTorch from the requirements file does not give you the setup
you need.

After training and evaluation, the final cells save the fine-tuned model and
launch the Gradio demo. Write a sentence about how you feel, select a
recommendation mode, and explore the emotion scores, songs, and quote.

## How it works

1. Loads the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)
   dataset, which contains English text labelled with emotions.
2. Cleans and normalizes text, then explores the dataset.
3. Trains a TF-IDF and logistic-regression baseline.
4. Fine-tunes `distilbert-base-uncased` for six-way emotion classification.
5. Evaluates the models and saves the fine-tuned classifier.
6. Uses the predicted mood to select curated songs and a quote, then presents
   the results in Gradio.

## A few things to keep in mind

This is a learning project, not a mental-health tool. The dataset is made up
of English tweets, and real feelings are more nuanced than six labels. The
recommendations are hand-curated examples, and model predictions can be wrong.
The notebook's own future-work section explores ideas like broader language
support, richer mood combinations, and personalized recommendations.

## Project contents

- [`nlp-project.ipynb`](./nlp-project.ipynb) — data exploration, training,
  evaluation, recommendations, and the interactive demo.

# %% [markdown]
# # CRAFT fine-tuning and inference interactive demo
# 
# This example notebook shows how to fine-tune a pretrained CRAFT conversational model for the task of forecasting conversational derailment, as shown in the "Trouble on the Horizon" paper (note however that due to nondeterminism in the training process, the results will not exactly reproduce the ones shown in the paper; if you need the exact inference results from the paper, see our [online demo](https://colab.research.google.com/drive/1GvICZN0VwZQSWw3pJaEVY-EQGoO-L5lH) that does inference only using the saved already-fine-tuned model from the paper).
# 
# Also note that this notebook is written primarily for the Wikipedia data. It will still work on the Reddit CMV data as well, but be aware that if seeking to compare results to those in the paper, the actual Reddit CMV evaluation contains some nuances that are not present in the Wikipedia data, as detailed in the [CMV version of the online demo](https://colab.research.google.com/drive/1aGBUBeiF3jT-GtBU9SDUoxhsjwKZaMKl?usp=sharing).

# %%
# Set before importing pytorch: https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018/12.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# %%
# import necessary libraries, including convokit
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import random
import unicodedata
import itertools
from convokit import download, Corpus
from tqdm import tqdm
from sklearn.metrics import roc_curve

from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import evaluate

from config import *

ROOT_DIR = os.path.join("/reef/bert_variance/", corpus_name)
RUNS_DIR = os.path.join(ROOT_DIR, "runs")
existing_runs = os.listdir(RUNS_DIR)
if len(existing_runs) == 0:
    cur_run = 0
else:
    runs = [int(r) for r in existing_runs]
    cur_run = np.max(runs) + 1
save_dir = os.path.join(RUNS_DIR, str(cur_run))
print("using path", save_dir)
os.mkdir(save_dir)



# %% [markdown]
# ## Part 1: set up data preprocessing utilities
# 
# We begin by setting up some helper functions for preprocessing the ConvoKit Utterance data for use with CRAFT.

# %%
# Given a ConvoKit conversation, preprocess each utterance's text by tokenizing and truncating.
# Returns the processed dialog entry where text has been replaced with a list of
# tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
def processDialog(dialog):
    processed = []
    for utterance in dialog.iter_utterances():
        # skip the section header, which does not contain conversational content
        if corpus_name == 'wikiconv' and utterance.meta['is_section_header']:
            continue
        processed.append({"text": utterance.text, "is_attack": int(utterance.meta[utt_label_metadata]) if utt_label_metadata is not None else 0, "id": utterance.id})
    if utt_label_metadata is None:
        # if the dataset does not come with utterance-level labels, we assume that (as in the case of CMV)
        # the only labels are conversation-level and that the actual toxic comment was not included in the
        # data. In that case, we must add a dummy comment containing no actual text, to get CRAFT to run on 
        # the context preceding the dummy (that is, the full prefix before the removed comment)
        processed.append({"text": "", "is_attack": int(dialog.meta[label_metadata]), "id": processed[-1]["id"] + "_dummyreply"})
    return processed

# Load context-reply pairs from the Corpus, optionally filtering to only conversations
# from the specified split (train, val, or test).
# Each conversation, which has N comments (not including the section header) will
# get converted into N-1 comment-reply pairs, one pair for each reply 
# (the first comment does not reply to anything).
# Each comment-reply pair is a tuple consisting of the conversational context
# (that is, all comments prior to the reply), the reply itself, the label (that
# is, whether the reply contained a derailment event), and the comment ID of the
# last comment in the context (for later use in re-joining with the ConvoKit corpus).
# The function returns a list of such pairs.
def corpus2dataset(corpus, split=None, last_only=False, shuffle=False):
    dataset_dict = {
        "context": [],
        "id": [],
        "labels": []
    }
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        if split is None or convo.meta['split'] == split:
            dialog = processDialog(convo)
            iter_range = range(1, len(dialog)) if not last_only else [len(dialog)-1]
            for idx in iter_range:
                label = dialog[idx]["is_attack"]
                # when re-joining with the corpus we want to store forecasts in
                # the last comment of each context (i.e. the comment directly
                # preceding the reply), so we must save that comment ID.
                comment_id = dialog[idx-1]["id"]
                # gather as context all utterances preceding the reply
                context = [u["text"] for u in dialog[:idx]]
                dataset_dict["context"].append(context)
                dataset_dict["id"].append(comment_id)
                dataset_dict["labels"].append(label)
    if shuffle:
        return Dataset.from_dict(dataset_dict).shuffle(seed=2024)
    else:
        return Dataset.from_dict(dataset_dict)

# %% [markdown]
# ## Part 2: load the data
# 
# Now we load the labeled corpus (Wikiconv or Reddit CMV) from ConvoKit, and run some transformations to prepare it for use with PyTorch

# %%
if corpus_name == "wikiconv":
    corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
elif corpus_name == "cmv":
    corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))

# %%
# let's check some quick stats to verify that the corpus loaded correctly
print(len(corpus.get_utterance_ids()))
print(len(corpus.get_speaker_ids()))
print(len(corpus.get_conversation_ids()))

# %%
# Let's also take a look at some example data to see what kinds of information/metadata are available to us
print(list(corpus.iter_conversations())[0].__dict__)
print(list(corpus.iter_utterances())[0])

# %%
# load the corpus into PyTorch-formatted train, val, and test datasets
dataset = DatasetDict({
    "train": corpus2dataset(corpus, "train", last_only=True, shuffle=True), 
    "val": corpus2dataset(corpus, "val", last_only=True),
    "test": corpus2dataset(corpus, "test")
})

# %%
# check data sizes


# %%
# check examples to verify that processing happened correctly
for i in range(5):
    print(dataset["train"][i]["context"])

# %%
# tokenize the dataset so it is usable by huggingface
# shamelessly stolen from https://github.com/CornellNLP/calm (thanks tushaar)

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-cased", model_max_length=512, truncation_side="left", padding_side="right"
)

tokenizer_helper = lambda inst: tokenizer.encode_plus(
    text=f" {tokenizer.sep_token} ".join(inst["context"]), 
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    max_length=512,
)
tokenized_dataset = dataset.map(tokenizer_helper, remove_columns=["context"], num_proc=20)

# %%
tokenized_dataset

# %%
# Check to see if everything looks alright.
print(
    f"CONVO={tokenizer.decode(tokenized_dataset['train'][0]['input_ids'])}"
    f"\n\nLABEL={tokenized_dataset['train'][0]['labels']}"
)

# %%
tokenized_dataset.set_format("torch")

# %% [markdown]
# ## Part 3: define the inference pipeline
# 
# CRAFT inference consists of three steps: (1) using the utterance encoder to produce embeddings of each comment in the context (2) running the comment embeddings through the context encoder to get a final representation of conversational context (3) running the classifier head on the context embedding. To streamline the subsequent code, we encapsulate these three steps in a single PyTorch `nn.Module`.

# %% [markdown]
# ## Part 4: define training loop
# 
# Now that we have all the model components defined, we need to define the actual training procedure. This will be a fairly standard neural network training loop, iterating over batches of labeled dialogs and computing cross-entropy loss on the predicted label. We will also define evaluation functions so that we can compute accuracy on the validation set after every epoch, allowing us to keep the model with the best validation performance. Note that for the sake of simpler code, validation accuracy is computed in the "unfair" manner using a single run of CRAFT over the full context preceding the actual personal attack, rather than the more realistic (and complicated) iterated evaluation that is used for final evaluation of the test set (in practice the two metrics track each other fairly well, making this a reasonable simplification for the sake of easy validation).

# %% [markdown]
# ## Part 5: define the evaluation procedure
# 
# We're almost ready to run! The last component we need is some code to evaluate performance on the test set after fine-tuning is completed. This evaluation should use the full iterative procedure described in the paper, replicating how a system might be deployed in practice, without knowledge of where the personal attack occurs

# %%
@torch.inference_mode
@torch.no_grad
def evaluateDataset(dataset, finetuned_model, device, threshold=0.5, temperature=1.0):
    finetuned_model = finetuned_model.to(device)
    convo_ids = []
    preds = []
    scores = []
    for data in tqdm(dataset):
        input_ids = data['input_ids'].to(device, dtype = torch.long).reshape([1,-1])
        attention_mask = data['attention_mask'].to(device, dtype = torch.long).reshape([1,-1])
        outputs = finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits / temperature, dim=-1)
        convo_ids.append(data["id"])
        raw_score = probs[0,1].item()
        preds.append(int(raw_score > threshold))
        scores.append(raw_score)
    return pd.DataFrame({"prediction": preds, "score": scores}, index=convo_ids)

# %% [markdown]
# ## Part 6: build and fine-tune the model
# 
# We finally have all the components we need! Now we can instantiate the CRAFT model components, load the pre-trained weights, and run fine-tuning.

# %%
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)


# %%
def compute_metrics(eval_pred):
    cls_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return cls_metrics.compute(predictions=predictions, references=labels)

# %%
training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=6.7e-6,  # https://arxiv.org/pdf/2110.05111.pdf
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    prediction_loss_only=False,
    run_name=f"bertcraft_{corpus_name}",
    logging_steps=1,
    seed=4300,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [markdown]
# ## Part 6.5: Threshold tuning
# 
# For CRAFT, we selected the decision threshold by iterating over possible thresholds on the validation set and picking the one that gives the highest validation accuracy. Since the optimal threshold may be different for BERT, we replicate this process with our trained BERT forecaster.

# %%
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")

if corpus_name == 'wikiconv':
    checkpoint = "237"  # ep0: 79, ep1: 158, ep3: 237
elif corpus_name == 'cmv':
    checkpoint = "387"  # ep0: 129, ep1: 258, ep3: 387
# for custom data, specify your own checkpoint!
finetuned_model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(save_dir, f"checkpoint-{checkpoint}")
)


# %% [markdown]
# ## Part 7: run test set evaluation
# 
# Now that we have successfully fine-tuned the model, we run it on the test set so that we can evaluate performance.

# %%
forecasts_df = evaluateDataset(tokenized_dataset["test"], finetuned_model, "cuda")


# %% [markdown]
# ## Part 8: merge predictions back into corpus and evaluate
# 
# Now that the hard part is done, all that is left to do is to evaluate the predictions. Since the predictions are in no particular order, we will first merge each prediction back into the source corpus, and then evaluate each conversation according to the order of utterances within that conversation.

# %%
# We will add a metadata entry to each test-set utterance signifying whether, at the time
# that CRAFT saw the context *up to and including* that utterance, CRAFT forecasted the
# conversation would derail. Note that in datasets where the actual toxic comment is
# included (such as wikiconv), we explicitly do not show that comment to CRAFT (since
# that would be cheating!), so that comment will not have an associated forecast.
for convo in corpus.iter_conversations():
    # only consider test set conversations (we did not make predictions for the other ones)
    if convo.meta['split'] == "test":
        for utt in convo.iter_utterances():
            if utt.id in forecasts_df.index:
                utt.meta['forecast_score'] = forecasts_df.loc[utt.id].score

# %%
# Finally, we can use the forecast-annotated corpus to compute the forecast accuracy.
# Though we have an individual forecast per utterance, ground truth is at the conversation level:
# either a conversation derails or it does not. Thus, forecast accuracy is computed as follows:
#   - True positives are cases that actually derail, for which the model made at least one positive forecast ANYTIME prior to derailment
#   - False positives are cases that don't derail but for which the model made at least one positive forecast
#   - False negatives are cases that derail but for which the model made no positive forecasts prior to derailment
#   - True negatives are cases that don't derail, for which the model made no positive forecasts
# Note that in the included datasets (wikiconv and cmv), by construction, all forecasts we obtained are forecasts made prior to derailment
# (since these datasets end right before or right at the toxic comment). This simplifies  the computation of forecast metrics as we now 
# do not need to explicitly consider when a forecast was made. But if you are using a custom dataset where conversations continue past
# the toxic comment, you will need to take that into account when evaluating.

conversational_forecasts_df = {
    "convo_id": [],
    "label": [],
    "score": [],
    "prediction": []
}

for convo in corpus.iter_conversations():
    if convo.meta['split'] == "test":
        conversational_forecasts_df['convo_id'].append(convo.id)
        conversational_forecasts_df['label'].append(int(convo.meta[label_metadata]))
        forecast_scores = [utt.meta['forecast_score'] for utt in convo.iter_utterances() if 'forecast_score' in utt.meta]
        conversational_forecasts_df['score'] = np.max(forecast_scores)
        conversational_forecasts_df['prediction'].append(int(np.max(forecast_scores) > 0.5))

conversational_forecasts_df = pd.DataFrame(conversational_forecasts_df).set_index("convo_id")

# %%
# in addition to accuracy, we can also consider applying other metrics at the conversation level, such as precision/recall
def compute_all_metrics(preds, labels):
    acc = (labels == preds).mean()
    tp = ((labels==1)&(preds==1)).sum()
    fp = ((labels==0)&(preds==1)).sum()
    tn = ((labels==0)&(preds==0)).sum()
    fn = ((labels==1)&(preds==0)).sum()
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 / (((tp + fp) / tp) + ((tp + fn) / tp))
    return {"Accuracy": acc, "Precision": p, "Recall": r, "FPR": fpr, "F1": f1}

results = compute_all_metrics(conversational_forecasts_df.prediction, conversational_forecasts_df.label)
print(results)
with open(os.path.join(save_dir, "results.json"), "w") as fp:
    json.dump(results, fp)

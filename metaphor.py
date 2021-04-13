#!/usr/bin/env python3

"""
Metaphor identification script.
"""

import math
import random
import argparse
import textwrap
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from conceptnet5.vectors import query
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-e", "--epochs", dest="max_epochs", type=int, default=30,
                    help="Maximum number of epochs to learn from. Default=30")
parser.add_argument("-r", "--regularization", dest="l2", type=float, default=0.001,
                    help="l2 regularization value to use. Default=0.001")
parser.add_argument("--no-droput", dest="dropout", action="store_false", help="Turn off dropout layers")
parser.add_argument("-p", "--patience", dest="patience", type=int, default=3,
                    help="EarlyStopping patience parameter, -1 to disable early stopping. Default=3")
parser.add_argument("-o", "--output", dest="outdir", default="./output", help="Output directory")
parser.add_argument("-v", "--vectors", dest="vector_filename", default=None,
                    help="Location of vector file. Default=conceptnet5 data locations")
parser.add_argument("-c", "--corpus", dest="corpus_filename", default="./data/VUAMC.xml",
                    help='Location of the corpus to load. Default="./data/VUAMC.xml"')
args = parser.parse_args()

for name, value in vars(args).items():
    print(f"{name}: {value}")


def load_vuamc(url):
    """
    Loads the vuamc corpus and returns a list of the form (lematized sentence, metaphorical?)
    """
    def parse_sentence(sentence):
        """
        Takes a bs4 sentence tag and returns the lematized sentence and whether it was metaphorical
        """
        def is_metaphor(tag):
            return tag.name == "seg" and tag.get("function") in ["mrw", "mFlag"]  # and tag.get("subtype") != "WIDLII"

        lemmas = [w["lemma"].replace(" ", "_") for w in sentence.find_all("w")]
        # only look for words actually related to metaphor
        metaphorical = sentence.find(is_metaphor) is not None
        return lemmas, metaphorical

    if url.startswith("http"):
        document = requests.get(url).text
    else:
        with open(url, "r", encoding="utf8") as f:
            document = f.read()
    xml = BeautifulSoup(document, features="lxml")
    sentences = xml.find_all("s")
    sentences = map(parse_sentence, sentences)
    return list(sentences)


def split_labels(lst):
    """
    Takes a list of tuples of the form (input, label)
    Returns 2 lists, one of inputs and one of labels
    """
    return tuple(map(np.array, zip(*lst)))


def vectorize(sentence, vectors):
    sentence = " ".join(sentence)
    return vectors.text_to_vector("en", sentence)


def preprocessing(corpus, vectors):
    lst = [(vectorize(sentence, vectors), float(label)) for sentence, label in corpus]
    return np.array(lst)


# load numberbatch vectors
print("Loading vectors...")
vectors = query.VectorSpaceWrapper(vector_filename=args.vector_filename)
vectors.load()

# Load metaphor corpus
print("Loading corpus...")
corpus = load_vuamc(args.corpus_filename)
print("Preprocessing...")
# vectorize sentences and convert boolean labels to floats
labelled_data = preprocessing(corpus, vectors)

random.Random(10).shuffle(corpus)  # shuffle corpus for splitting
t_split = len(corpus) * 6 // 10
v_split = len(corpus) * 8 // 10

train_data, train_labels = split_labels(labelled_data[:t_split])
val_data, val_labels = split_labels(labelled_data[t_split:v_split])
test_data, test_labels = split_labels(labelled_data[v_split:])
test_labels_bool = test_labels.astype(dtype=bool)


def MetaphorModel():
    # regularizers and dropout layers help prevent overfitting
    input = layers.Input(shape=(300,))
    x = layers.Dense(300, activation="relu", kernel_regularizer=l2(args.l2), bias_regularizer=l2(args.l2))(input)
    if args.dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Dense(60, activation="relu", kernel_regularizer=l2(args.l2), bias_regularizer=l2(args.l2))(x)
    if args.dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation="sigmoid")(x)  # prediciton layer

    model = keras.Model(inputs=input, outputs=x)
    return model


model = MetaphorModel()
model.summary()

callbacks = []
# early stopping stops iteration when the model starts overfitting too much
if args.patience >= 0:
    callbacks.append(EarlyStopping(patience=args.patience, restore_best_weights=True))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_data, train_labels, steps_per_epoch=30, epochs=args.max_epochs,
                    validation_data=(val_data, val_labels), callbacks=callbacks)

# setup output directory
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# export accuracy plot
for vals in [history.history["accuracy"], history.history["val_accuracy"]]:
    plt.plot(range(1, len(vals)+1), vals)
plt.title("model accuracy")
plt.xlim(1, args.max_epochs)
plt.xlabel("epoch")
plt.ylim(0.5, 1.0)
plt.ylabel("accuracy")
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig(outdir / "model_accuracy.png")
plt.close()

# export loss plot
for vals in [history.history["loss"], history.history["val_loss"]]:
    plt.plot(range(1, len(vals)+1), vals)
plt.title("model loss")
plt.xlim(1, args.max_epochs)
plt.xlabel("epoch")
plt.ylim(0.0, 1.0)
plt.ylabel("loss")
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig(outdir / "model_loss.png")
plt.close()

# get predictions and metrics
predictions = model.predict(test_data)
predictions = predictions.reshape((len(predictions),))
predictions_bool = [pred >= 0.5 for pred in predictions]
test_metrics = {
    "accuracy": metrics.accuracy_score(test_labels_bool, predictions_bool),
    "precision": metrics.precision_score(test_labels_bool, predictions_bool),
    "recall": metrics.recall_score(test_labels_bool, predictions_bool),
    "f1": metrics.f1_score(test_labels_bool, predictions_bool),
}
eval_loss, eval_acc = model.evaluate(val_data, val_labels)

# export test metrics and predictions
num_len = math.ceil(math.log10(len(predictions)))
with open(outdir / "metrics_and_predictions.md", "w", newline="\n") as f:
    f.write(textwrap.dedent(f"""
        # Test
        Accuracy  = {test_metrics['accuracy']}
        Precision = {test_metrics['precision']}
        Recall    = {test_metrics['recall']}
        F1        = {test_metrics['f1']}
        # Eval
        Loss      = {eval_loss}
        Accuracy  = {eval_acc}
        ![Model Loss](model_loss.png)
        ![Model Accuracy](model_accuracy.png)
        # Preictions
        """).lstrip())
    for i, (actual, pred) in enumerate(zip(test_labels_bool, predictions)):
        f.write(f"{i:0{num_len}}: {str(actual):<5} vs {str(pred >= 0.5):<5} = {pred}\n")

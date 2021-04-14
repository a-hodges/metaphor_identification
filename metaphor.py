#!/usr/bin/env python3

"""
Metaphor identification using tensorflow deep learning.
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
                    help="l2 regularization value to use. 0.0 for no regularization. Default=0.001")
parser.add_argument("-d", "--droput", dest="dropout", type=float, default=0.5,
                    help="Dropout rate on intermediate dropout layers. 0.0 for no dropout. Default=0.5")
parser.add_argument("-p", "--patience", dest="patience", type=int, default=3,
                    help="EarlyStopping patience parameter, -1 to disable early stopping. Default=3")
parser.add_argument("-o", "--output", dest="outdir", default="./output", help="Output directory")
parser.add_argument("-v", "--vectors", dest="vector_filename", default=None,
                    help="Location of vector file. Default=conceptnet5 data locations")
parser.add_argument("-c", "--corpus", dest="corpus_filename", default="./data/VUAMC.xml",
                    help='Location of the corpus to load. Default="./data/VUAMC.xml"')
parser.add_argument("-l", "--language", dest="lang", default="en",
                    help='Language of the corpus. Should be a language code supported by ConceptNet5. Default="en"')

# Preprocessing functions


def load_numberbatch(filename):
    print("Loading vectors...")
    vectors = query.VectorSpaceWrapper(vector_filename=filename)
    vectors.load()
    return vectors


def load_vuamc(url):
    """
    Loads the vuamc corpus and returns a list of the form (lematized sentence, metaphorical?)
    """
    def parse_sentence(sentence):
        """
        Takes a bs4 sentence tag and returns the lematized sentence and whether it was metaphorical
        """
        def is_metaphor(tag):
            return tag.name == "seg" and tag.get("function") in ["mrw", "mFlag"] and tag.get("subtype") != "WIDLII"

        lemmas = [w["lemma"].replace(" ", "_") for w in sentence.find_all("w")]
        # only look for words actually related to metaphor
        metaphorical = sentence.find(is_metaphor) is not None
        return lemmas, metaphorical

    print("Loading corpus...")

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
    return tuple(zip(*lst))


def vectorize(vectors, lang, sentence):
    sentence = " ".join(sentence)
    return vectors.text_to_vector(lang, sentence)


def preprocess(corpus, vectors, lang="en"):
    data, labels = split_labels(corpus)
    data = np.array([vectorize(vectors, lang, s) for s in data])
    labels = np.array(list(map(float, labels)))
    return data, labels


def preprocessing(corpus, vectors, lang="en"):
    print("Preprocessing...")

    # shuffle corpus for splitting
    random.Random(10).shuffle(corpus)
    t_split = len(corpus) * 6 // 10
    v_split = len(corpus) * 8 // 10

    # vectorize sentences and convert boolean labels to floats
    train = preprocess(corpus[:t_split], vectors, lang=lang)
    val = preprocess(corpus[t_split:v_split], vectors, lang=lang)
    test = preprocess(corpus[v_split:], vectors, lang=lang)
    return train, val, test


# model functions


def MetaphorModel(l2s, dropout_rate):
    # regularizers and dropout layers help prevent overfitting
    input = layers.Input(shape=(300,))
    x = layers.Dense(300, activation="relu", kernel_regularizer=l2(l2s), bias_regularizer=l2(l2s))(input)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(300, activation="relu", kernel_regularizer=l2(l2s), bias_regularizer=l2(l2s))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(60, activation="relu", kernel_regularizer=l2(l2s), bias_regularizer=l2(l2s))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1, activation="sigmoid")(x)  # prediciton layer

    model = keras.Model(inputs=input, outputs=x)
    return model


def run_model(model, patience, max_epochs, train, val=None):
    train_data, train_labels = train

    callbacks = []
    # early stopping stops iteration when the model starts overfitting too much
    if patience >= 0:
        callbacks.append(EarlyStopping(patience=patience, restore_best_weights=True))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, steps_per_epoch=30, epochs=max_epochs,
                        validation_data=val, callbacks=callbacks)
    return model, history


# output functions


def plot_accuracy(path, history, max_epochs=None):
    for vals in [history.history["accuracy"], history.history["val_accuracy"]]:
        plt.plot(range(1, len(vals)+1), vals)
    plt.title("model accuracy")
    plt.xlim(1, max_epochs)
    plt.xlabel("epoch")
    plt.ylim(0.5, 1.0)
    plt.ylabel("accuracy")
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def plot_loss(path, history, max_epochs=None):
    for vals in [history.history["loss"], history.history["val_loss"]]:
        plt.plot(range(1, len(vals)+1), vals)
    plt.title("model loss")
    plt.xlim(1, max_epochs)
    plt.xlabel("epoch")
    plt.ylim(0.0, 1.0)
    plt.ylabel("loss")
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def save_metrics(path, model, val, test=None):
    if test is None:
        test = val

    test_data, test_labels = test
    test_labels_bool = test_labels.astype(dtype=bool)

    predictions = model.predict(test_data)
    predictions = predictions.reshape((len(predictions),))
    predictions_bool = [pred >= 0.5 for pred in predictions]

    test_metrics = {
        "accuracy": metrics.accuracy_score(test_labels_bool, predictions_bool),
        "precision": metrics.precision_score(test_labels_bool, predictions_bool),
        "recall": metrics.recall_score(test_labels_bool, predictions_bool),
        "f1": metrics.f1_score(test_labels_bool, predictions_bool),
    }

    eval_loss, eval_acc = model.evaluate(*val)

    # export test metrics and predictions
    num_len = math.ceil(math.log10(len(predictions)))
    with open(path, "w", newline="\n") as f:
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
    print(f"Saved {path}")


# main


def main():
    args = parser.parse_args()
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    vectors = load_numberbatch(args.vector_filename)
    corpus = load_vuamc(args.corpus_filename)
    train, val, test = preprocessing(corpus, vectors, args.lang)

    model = MetaphorModel(l2s=args.l2, dropout_rate=args.dropout)
    model.summary()
    model, history = run_model(model, patience=args.patience, max_epochs=args.max_epochs, train=train, val=val)

    # setup output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_accuracy(outdir / "model_accuracy.png", history, args.max_epochs)
    plot_loss(outdir / "model_loss.png", history, args.max_epochs)
    save_metrics(outdir / "metrics_and_predictions.md", model, val, test)


if __name__ == "__main__":
    main()

"""Course-provided skeleton for Part A: CRF slot filling.

The constructor, `fit()`, and `predict()` are fully implemented — they
wire the model together and call into `sklearn-crfsuite`. The one thing
left to implement is `extract_features()`: turning a tokenized sentence
into a per-token feature dict. Per the README, feature engineering *is*
the Part A assignment — look for the `# TODO: implement` block below.
"""

from __future__ import annotations
from pyexpat import features
from typing import Any, cast
import sklearn_crfsuite


class CRFModel:
    def __init__(self) -> None:
        """Initialize underlying `sklearn-crfsuite` CRF estimator.

        Reasonable defaults are set here; override them by editing this
        constructor or by reconstructing `self.model` with different
        hyperparameters as you experiment.
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )

    def extract_features(self, sentence: list[str]) -> list[dict[str, Any]]:
        """Convert a tokenized sentence into one feature dict per token.

        This is the part you implement.

        Args:
            sentence: A list of token strings, e.g. ["show", "me", "flights", "to", "boston"].

        Returns:
            A list the same length as `sentence`, where each element is a
            dict of features for the token at that position. Typical
            features to consider: word identity, casing, prefixes/suffixes,
            surrounding-token context window, gazetteers if you build them.
        """
        # ------------------------------------------------------------------
        # TODO: implement
        # ------------------------------------------------------------------
        features = []

        for i, word in enumerate(sentence):
            # Example feature: the word itself
            feature_dict = {
                "word": word,
                # E1 Casing and Morphology
                "word_lower": word.lower(),
                "is_upper": word.isupper(),
                "is_title": word.istitle(),
                "is_digit": word.isdigit(),
                # E2 prefix and suffix
                "prefix_2": word[:2],
                "prefix_3": word[:3],
                "suffix_2": word[-2:],
                "suffix_3": word[-3:],
            }

            # E3 Previous / Next Word Shape

            if i > 0:
                prev_word = sentence[i - 1]
                feature_dict.update({
                    "prev_word": prev_word,
                    "prev_word_lower": prev_word.lower(),
                })
            else:
                feature_dict["BOS"] = True  # Beginning of sentence

            if i < len(sentence) - 1:
                next_word = sentence[i + 1]
                feature_dict.update({
                    "next_word": next_word,
                    "next_word_lower": next_word.lower(),
                })
            else:
                feature_dict["EOS"] = True  # End of sentence

            # E4: Wider Context Window

            if i > 1:
                prev2_word = sentence[i - 2]

                feature_dict.update({
                    "prev2_word": prev2_word,
                    "prev2_word_lower": prev2_word.lower(),
                })

            if i < len(sentence) - 2:
                next2_word = sentence[i + 2]

                feature_dict.update({
                    "next2_word": next2_word,
                    "next2_word_lower": next2_word.lower(),
                })

            features.append(feature_dict)

        return features

    def fit(self, sentences: list[list[str]], labels: list[list[str]]) -> None:
        """Train CRF on tokenized sentences and their BIO label sequences.

        Args:
            sentences: A list of tokenized sentences.
            labels: A list of BIO label sequences, one per sentence, aligned
                token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        self.model.fit(features, labels)

    def predict(self, sentences: list[list[str]], vocab: Any = None) -> list[list[str]]:
        """Predict BIO label sequences for tokenized sentences.

        Args:
            sentences: A list of tokenized sentences.
            vocab: Unused. Accepted only so `CRFModel.predict()` and
                `BiLSTMModel.predict()` share the exact same call signature —
                sklearn-crfsuite works on raw token features, not vocab ids,
                so there's nothing for CRF to look up here. This is
                intentional, not a bug.

        Returns:
            A list of predicted BIO label sequences, one per sentence,
            aligned token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        return cast(list[list[str]], self.model.predict(features))

import os.path
from pathlib import Path

# get the absolute path to the repository so we don't have to deal with relative paths
repo_dir = Path(__file__).parent.parent.absolute()

corpus_name = "cmv" # Name of the dataset to run CRAFT on. This is not directly used by the model, it is instead used by
                         # this config file to define certain input and output locations. You can, of course, override those
                         # location settings directly and thus completely ignore this setting, it is just useful to use this
                         # setting to keep everything consistent and organized :)
                         # Note that in addition to the default setting of "wikiconv" you can also set this to "cmv" and still
                         # have the code work out-of-the-box (with Reddit CMV data) as the repo includes all support files needed
                         # for both the Wikiconv and Reddit CMV corpora.
# Name of the conversation metadata field in the ConvoKit corpus to use as the label for training and evaluation.
# Note that the if-statement in the default value is only there to enable users to switch to the CMV data by only
# changing the corpus_name (rather than having to directly change label_metadata as well). If you are using a custom
# corpus, then of course the if-statement is not needed and you can just directly put the name of the metadata
# field in your corpus that you want to use as the label.
label_metadata = "conversation_has_personal_attack" if corpus_name == "wikiconv" else "has_removed_comment"
# Name of the utterance metadata field that contains comment-level toxicity labels, if any. Note
# that CRAFT does not strictly need such labels, but some datasets like the wikiconv data do include
# it. For custom datasets it is fine to leave this as None.
utt_label_metadata = "comment_has_personal_attack" if corpus_name == "wikiconv" else None

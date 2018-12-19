# NER-using-HMMs-MEMMs

Goal: The goal of this project was to attempt to apply the BIO tagging scheme for Named Entity Recognition using Hidden Markov Models and Maximum Ent Markov Models.

Baseline Systems
We structured two baseline systems off of the example given in Section 3.3 of the
assignment instructions. If a word was unknown, we assigned it an “O.” The two systems differ
as follows:

Baseline 1: If the word appears in the lexicon, select the tag that was associated with that
word most often in the training corpus (i.e. If the word “Washington” appeared 3 times in the
training corpus, twice as B-LOC and once as I-PER, then “Washington” would always be labeled
as B-LOC in testing).

Baseline 2: If the word appears in the lexicon, randomly select any tag that was
associated with that word in the training corpus (i.e. If “Washington” appeared in the training
corpus as B-LOC and I-PER, then either B-LOC or I-PER would be randomly selected as the tag
for “Washington” during testing).

Our motivation for creating these two baselines with the difference in how the tag is
selected during testing was to measure whether it is better to select the most common tag or to
randomly select any tag that was associated with a particular word. In addition, this type of
baseline allowed us to attempt to identify every known word based on the training data; this
allowed us to compare our HMM and MEMM models that also incorporated unknown handling
to models that were at least able to “guess” a tag for every seen word.

In our preprocessing (more on this in the next section of the report), we created several
dictionaries to keep track of the words, tags, and relationships between them. For our lexicon, we
used a dictionary of dictionaries that stored each word as a key and a dictionary (with the tags
associated with that word as keys and number of occurrences of each tag with that word as
values) as a value.

HMMs:

For HMMs, we first needed the transition probabilities between tags and the probabilities
of the words given the tag. In order to achieve these two sets of values, we used 2 dictionaries
created during pre-processing: train_tag_bigram and train_words. Both of these will be explained
further in the pre-processing section, but train_tag_bigram is a dictionary of dictionaries
containing the counts for every tag that follows a given tag. Likewise, train_words is a dictionary
of dictionaries containing the counts for every tag seen with a given word. To get the
probabilities, I iterated through each of these dictionaries and divided the counts for every single
pair of tags/word-tag pairs by the number of times the first tag/word appeared.
In order to adjust our HMM result on our validation set, we attempted to use both
Good-Turing smoothing and Add-K smoothing. We originally used Good-Turing smoothing, but
there was such minimal impact on our results <0.001, we switched over to add-k smoothing. For
the Add-K smoothing, we used various k scores in order to smooth the bigram probabilities of
tags given tags and the probabilities of words given tags. We intuitively thought it made sense to
use different k values, because the weight of adding hallucinated counts would have a larger
impact on the probabilities of words given tags because there are far less counts of each word
than counts of each tag. As a result, we ended up with k-values of 2 and 0.0001 for the tag
bigram and word-tag dictionary, respectively.

To deal with unknown words, we added a condition during our pre-processing where
there was a 30% chance (using numpy’s random feature) to replace a unique word (never seen in
our dictionary array) with an <UNK> tag. This <UNK> tag persisted throughout the rest of the
process and just acted as another word. During our validation/testing phases, we went through
the validation set and test set and replaced words that were not in our dictionary with these
<UNK> tags.
  
We primarily used the Viterbi algorithm in order to predict the tags for each line in the
testing set. Using both the transition probabilities and the probabilities of words given tags, we
processed both our validation and testing sets and enacted the Viterbi algorithm on each line for
both sets, ultimately saving the resulting tags in a larger array.

For the algorithm, we initialize a score matrix, containing the scores/probabilities for the
various tag categories and words and a backpointer array, containing the score maximizing
previous index, in nxd matrices (n NER tags and d words in a line). Going through each word in
the line, we simply added the probabilities to the score matrix that maximized the probability of
the previous score*probability of the current tag given previous tag and multiplied that by the
probability of the current word given by the current tag. Throughout this process, we also added
the maximizing indices to our backpointer matrix. After filling our score matrix, we backtracked
using the backpointer array to construct the maximum score/probability sequence of words and
finally returned that in an array.

MEMMs
We used the following features for our MEMM implementation. We realized that using the
word’s context was really important to correctly tag the current word.
For the current, previous word, next word:
- Checked if part of speech was NNP or NNPS
- Length of word
- If first letter is capitalized
- If entire word is capitalized
- If word contains hyphen
- If word has non alphabetical characters
- Checked word shape

For just the previous word, next word:
- BIO tag
- Checked if part of speech was IN, JJ, VB, VBD, POS, or RB
- If word = “,”

Reasoning behind using these features:
- Part of speech: we felt that part of speech would be a good indicator of each BIO tag
since we know that if the part of speech was NNP or NNPs, then it would be a B or I tag
with high probability
- Length of word: short words are most likely not named entities, and very long words are
likely named entities, such as a long last name.
- First letter capitalized: if the first letter is capitalized, then it is likely a person’s name,
location, or organization
- Entire word is capitalized: if the whole word is capitalized, then the word might be an
acronym, which could be a named entity
- If previous word is “,”: the comma could be between a city, state named entity
- If word contains hyphen: words with hyphens could represent dates
- If word has non alphabetical characters: words with numbers and symbols could also
represent dates
- Word shape: gives a hint as to the type of word, especially if it contains numbers or is an
acronym

This viterbi algorithm was executed in the same way as the HMM implementation with a
few modifications. Unlike HMM where we used train_tag_bigram_prob and train_words_prob
(the dictionaries holding the various probabilities), we simply used the probability distributions
for each word given to us through the maxEnt classifier. So, instead of maximizing the previous
word’s score*both sets of probabilities, we just maximized the previous word’s score*probability
of the BIO tag given the word. We used similar data structures as HMM to incorporate the
algorithm and to backtrack.

To implement MEMM, we looped through the data by each word from each line. Each
word had a feature vector that was stored as a list. We used the sklearn maxEnt classifier to
calculate the probability of a BIO tag given that word. We then ran these probabilities through
our viterbi algorithm, similar to our HMM implementation.

We tried using the GloVe word embedding for the current and previous words. However,
we ended up with an excessive number of features, and we eventually decided not to add features
from word embeddings because we believed that the other more important and distinguishing
features listed above would get washed out by the excessive number of features from the word
embeddings.

Results:
Our HMM had the highest accuracy among all of our models including MEMM. We reason it's primarily because of the features we decided to use.

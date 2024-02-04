# French DSL-ML data

## Data Format

The training data contains the following files:
```
	train.txt.gz - training set (gzip-compressed due to GitHub file size limitations)
	train.labels - training labels
	dev.txt - development/validation set
	dev.labels - development/validation labels
```

The `*.txt` files contain the data samples (one example per row):
```
	text-sample-1
	text-sample-2
	text-sample-3
	...
```

The `*.labels` files contain the labels (one or several labels per row, comma-separated):
```
	dialect-label-1
	dialect-label-2a,dialect-label2b
	dialect-label-3
	...
```

## Task Description

In the 2024 Multi-label French Dialect Identification (MFDI) shared task, participants have to train a model on news samples collected from a set of publication sources and evaluate it on news samples collected from a sub set of the publication sources. Although the sources not are different, the topics are different. Therefore, participants have to build a model for a cross-topic multi-label classification by dialect task, in which a classification model is required to discriminate between the French (FH), Swiss (CH), Belgian (BE) and Canadian (CA) dialects across different news samples. 

For the shared task, we provide participants with a combination of the FreCDo [1] and DSLCC data sets [2] which contain French (FR-FR), Swiss (FR-CH), Belgian (FR-BE) and Canadian (FR-CA) samples of text collected from the news domain. The corpus is divided into training, validation and test, such that the topics are distinct across splits. The training set contains 340,363 samples. The development set is composed of 17,090 samples. Another set of about 4,000 samples are kept for the final evaluation. All samples are preprocessed in order to replace named entities with a special tag: $NE$.

## Data Sources / References

[1] Mihaela Găman, Adrian Gabriel Chifu, William Domingues, Radu Tudor Ionescu. FreCDo: A New Corpus for Large-Scale French Cross-Domain Dialect Identification. Procedia Computer Science vol. 225, pp. 366-373, 2023. [FreCDo](https://github.com/MihaelaGaman/FreCDo)

[2] Liling Tan, Marcos Zampieri, Nikola Ljubešić, Jörg Tiedemann. Merging Comparable Data Sources for the Discrimination of Similar Languages: The DSL Corpus Collection. Proceedings of the 7th Workshop on Building and Using Comparable Corpora (BUCC), pp. 6-10, 2014. [DSLCC v4.0](http://ttg.uni-saarland.de/resources/DSLCC/)

## Submission Types

Participants are evaluated in two separate scenarios:
- CLOSED: participants are not allowed to use pre-trained models or external data to train their models.
- OPEN: participants are allowed to use external resources such as unlabeled corpora, lexicons, and pre-trained embeddings (e.g., BERT) but the use of additional labeled data is not allowed.

## Evaluation

The test data (to be released later) will only contain examples without their dialect labels. Participants will be required to submit the labels for these test instances. The exact details of the submission file format will be provided later.

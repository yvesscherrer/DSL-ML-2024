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

## Test Data Format

The test data contains the following files:

	test.txt - contain the test samples (one example per row):

	text-sample-1
	text-sample-2
	text-sample-3
	...

## Submission

Each participant is allowed to submit 3 runs for the CLOSED task and 3 runs for the OPEN task to:
	
	yves.scherrer@gmail.com 

Each line in the submission file MFDI-[task_type]-run-X-[team_name].txt file is in the format:

	label-1
	label-2a,label-2b
	...
	label-N

The labels must be given in the same order as the test samples listed in test.txt. The participants must provide labels for all the test samples. Each submission (run) must be accompanied by a MFDI-readme-[task_type]-run-X-[team_name].txt file containing a one-paragraph description of the respective submission, where X is the run number (1, 2 or 3). 

For example, if the team name is "Ghostbusters", the first submission to the CLOSED task should contain two files:
	MFDI-closed-run-1-Ghostbusters.txt
	MFDI-readme-closed-run-1-Ghostbusters.txt


## Evaluation

The macro-averaged F1 score will be used to rank the participants.

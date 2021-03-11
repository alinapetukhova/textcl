## TextCL
[![Build Status](https://travis-ci.com/alinapetukhova/textcl.svg?branch=master)](https://travis-ci.com/github/alinapetukhova/textcl)
[![codecov](https://codecov.io/gh/alinapetukhova/textcl/branch/master/graph/badge.svg?token=jgYuXyGGjS)](https://codecov.io/gh/alinapetukhova/textcl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Introduction
This package is aimed to clean text data for later usage in Natural Language Processing tasks. It can be used as an initial step in text analysis, predictive, classification, or text generation models. 

The quality of that models strongly depends on the quality of input data. Common problems in the data sets: 
- If data are coming from the Optical character recognition platform (OCR) the tables and columns formatted text is usually not processed correctly and will add noise to the models. 
- Some parts of the big texts scopes may contain sentences from different languages rather than the target language of the model and have to be filtered. 
- When we are working with real word texts we can see that some sentences can be duplicated due to the usage of templates. For the text generation tasks, it can cause the overfitting of the model and duplication in the created texts or summaries. 
- Data set contains texts, that are different from the main topic (like weather forecaset in the accounting reports)
---------------------------------
### List of functions

To be albe to solve pre-processing problems **TextCL** package includes following methods: 
- splitting texts into sentences
- filtering sentences by language (to remove sentences from text not in the target language)
- perplexity (to remove linguistically unconnected sentences, that can be produced by OCR modules. Example: *Sustainability Report 2019 36 3%?!353? 1. 5В°C 1} 33%.*)
- Jaccard similarity (to remove duplicated sentences in the text) 
- unsupervised outlier detection (to detect texts, that are outside of the main data set topic destribution). It can be applied using one of the modes: 
    - TONMF (block coordinate descent framework)
    [source article](https://arxiv.org/pdf/1701.01325.pdf),
    [matlab implementation](https://github.com/ramkikannan/outliernmf)
    - RPCA (Robust Principal Component Analysis)
    [source article](https://arxiv.org/pdf/0912.3599.pdf)
    [python implementation](https://github.com/dganguli/robust-pca)
    - Singular value decomposition: Used implementation from numpy: np.linalg.svd

    
---------------------------------
### Requirements

Python >= 3.6
- flair>=0.7
- langdetect>=1.0.8
- pandas>=1.0.3
- lxml>=4.6.2
- protobuf>=3.14.0
- nltk>=3.4.5

---------------------------------
### How to install

```bash
git clone -b text-processing https://[USER]:[PASS]@github.com/alinapetukhova/textcl.git
cd textcl
pip install src/
```

`src/` is path to the package folder where file 'setup.py' is located.

---------------------------------
### Usage examples

Load the text data you want to process. It's necessary to have column `text` in the data (default name). If you don't have `text` column you will need to specify the name for **split_into_sentences** function using `text_col` parameter. Source file from this example structured as follows:

|id| topic_name | text |
|:----|:----|:----|
| 0   |      business |  WorldCom bosses' $54m payout  Ten former direc...| 
| 1   |      business |  Profits slide at India's Dr Reddy  Profits at ...| 
| 2   |      business |  Liberian economy starts to grow  The Liberian ...| 
| 3   |      business |  Uluslararası Para Fonu (IMF), Liberya ekonomis...| 
| 4  |  entertainment |  Singer Ian Brown 'in gig arrest'  Former Stone...| 
| 5  |  entertainment |  Blue beat U2 to top France honour  Irish band ...| 
| 6  |  entertainment |  Housewives lift Channel 4 ratings  The debut o...| 
| 7  |  entertainment |  Домохозяйки подняли рейтинги канала 4 Дебют ам...| 
| 8  |  entertainment |  Housewives Channel 4 reytinglerini yükseltti A...| 
| 9  |       politics |  Observers to monitor UK election  Ministers wi...| 
| 10  |      politics |  Lib Dems highlight problem debt  People vulner...| 
| 11  |      politics |  Minister defends hunting ban law  The law bann...| 
| 12  |         sport |  Legendary Dutch boss Michels dies  Legendary D...| 
| 13  |         sport |  Connors boost for British tennis  Former world...| 
| 14  |         sport |  Sociedad set to rescue Mladenovic  Rangers are...| 
| 15  |          tech |  Mobile games come of age  The BBC News website...| 
| 16  |          tech |  PlayStation 3 processor unveiled  The Cell pro...| 
| 17  |          tech |  PC photo printers challenge printed...| 
| 18  |          tech |  PC photo printers challenge pros  Home printed...| 
| 19  |          tech |  Example 43 t6 43 Table data 342 5 3.4. data cl...| 
| 20  |          tech |  Janice Dean currently serves as senior meteoro...|

Import the package and pandas:

```Python
import textcl
import pandas as pd
```

Get data from file:

```Python
input_texts_df = pd.read_csv("prepared_bbc_dataset.csv")
```

Split input texts into sentences:

```Python
split_input_texts_df = textcl.split_into_sentences(input_texts_df)
```


##### Filtering on language
```Python
split_input_texts_df = textcl.language_filtering(split_input_texts_df, threshold=0.99, language='en')
```

Result:

|id| text |
|:----|:----|
| 0  | WorldCom bosses' $54m payout  Ten former direc...|
| 1  | Profits slide at India's Dr Reddy  Profits at ...|
| 2  | Liberian economy starts to grow  The Liberian ...|
| 4  | Singer Ian Brown 'in gig arrest'  Former Stone...|
| 5  | Blue beat U2 to top France honour  Irish band ...|
| 6  | Housewives lift Channel 4 ratings  The debut o...|
| 9  | Observers to monitor UK election  Ministers wi...|
| 10  | Lib Dems highlight problem debt  People vulner...|
| 11  | Minister defends hunting ban law  The law bann...|
| 12  | Legendary Dutch boss Michels dies  Legendary D...|
| 13  | Connors boost for British tennis  Former world...|
| 14  | Sociedad set to rescue Mladenovic  Rangers are...|
| 15  | Mobile games come of age  The BBC News website...|
| 16  | PlayStation 3 processor unveiled  The Cell pro...|
| 17  | PC photo printers challenge printed...|
| 18  | PC photo printers challenge pros  Home printed...|
| 19  | data clear additional 78.0 long-term 43 those)|
| 20  | Janice Dean currently serves as senior meteoro...|


As we can see texts with id 8 (Turkish), 7 (Russian), 8 (Turkish) were removed.

##### Filtering on Jaccard similarity

```Python
split_input_texts_df = textcl.jaccard_sim_filtering(split_input_texts_df, threshold=0.8)
```

Result:

|id| text |
|:----|:----|
| 0  | WorldCom bosses' $54m payout  Ten former direc...|
| 1  | Profits slide at India's Dr Reddy  Profits at ...|
| 2  | Liberian economy starts to grow  The Liberian ...|
| 4  | Singer Ian Brown 'in gig arrest'  Former Stone...|
| 5  | Blue beat U2 to top France honour  Irish band ...|
| 6  | Housewives lift Channel 4 ratings  The debut o...|
| 9  | Observers to monitor UK election  Ministers wi...|
| 10  | Lib Dems highlight problem debt  People vulner...|
| 11  | Minister defends hunting ban law  The law bann...|
| 12  | Legendary Dutch boss Michels dies  Legendary D...|
| 13  | Connors boost for British tennis  Former world...|
| 14  | Sociedad set to rescue Mladenovic  Rangers are...|
| 15  | Mobile games come of age  The BBC News website...|
| 16  | PlayStation 3 processor unveiled  The Cell pro...|
| 18  | PC photo printers challenge pros  Home printed...|
| 19  | data clear additional 78.0 long-term 43 those)|
| 20  | Janice Dean currently serves as senior meteoro...|

Texts with id=17 was removed as it partially duplicates text with id=18.

##### Filtering on perplexity score
```Python
split_input_texts_df = tp.perplexity_filtering(split_input_texts_df, threshold=5)
```

Result:

|id| text |
|:----|:----|
| 0  | WorldCom bosses' $54m payout  Ten former direc...|
| 1  | Profits slide at India's Dr Reddy  Profits at ...|
| 2  | Liberian economy starts to grow  The Liberian ...|
| 4  | Singer Ian Brown 'in gig arrest'  Former Stone...|
| 5  | Blue beat U2 to top France honour  Irish band ...|
| 6  | Housewives lift Channel 4 ratings  The debut o...|
| 9  | Observers to monitor UK election  Ministers wi...|
| 10  | Lib Dems highlight problem debt  People vulner...|
| 11  | Minister defends hunting ban law  The law bann...|
| 12  | Legendary Dutch boss Michels dies  Legendary D...|
| 13  | Connors boost for British tennis  Former world...|
| 14  | Sociedad set to rescue Mladenovic  Rangers are...|
| 15  | Mobile games come of age  The BBC News website...|
| 16  | PlayStation 3 processor unveiled  The Cell pro...|
| 18  | PC photo printers challenge pros  Home printed...|
| 20  | Janice Dean currently serves as senior meteoro...|

Texts with id=19 was removed because sentence `data clear additional 78.0 long-term 43 those)` is not linguistically correct.

##### Outliers filtering for categoty "tech"

```Python
joined_texts = split_input_texts_df[["text", "topic_name"]].drop_duplicates()
joined_texts, _ = textcl.outlier_detection(joined_texts[joined_texts.topic_name == 'tech'], method='rpca', Z_threshold=0.8)
```

Result:

|id| text |
|:----|:----|
| 15  | Mobile games come of age  The BBC News website...|
| 16  | PlayStation 3 processor unveiled  The Cell pro...|
| 18  | PC photo printers challenge pros  Home printed...|

Texts with id=20 was removed because it describes a person profile instead of tech news.


Also sentences grouped by topic can be joined to texts with the same number of sentences in them

```Python
# joining sentences into texts
joined_texts = textcl.join_sentences_by_topics(split_input_texts_df)
print("Num texts after joining: {}".format(len(joined_texts)))
```

After joining the result DataFrame will look like: 

| topic_name | joined_sentences |
|:----|:----|
| business | James Wareham, a lawyer representing one of t... |
| entertainment | Singer Ian Brown 'in gig arrest'  Former Stone... |
| politics | Observers to monitor UK election  Ministers wi... |
| sport | Referred to in the Netherlands as "the Genera... |
| tech | Mobile games come of age  The BBC News website... |


---------------------------------
### Developer's guide

To generate documentation use (it will be placed into the docs folder):

```bash
pdoc3 --html --output-dir docs src/textcl/
```

where `scr/textcl/` is a path to folder contains __init__.py  

Also documentation can be found [here](https://alinapetukhova.github.io/textcl/docs/).


--------------------------------
To run tests use in the root folder:

```bash
pytest
```

To check tests coverage use:

```bash
pytest --cov=textcl --cov-report=html
```

### License

[MIT License](LICENSE)
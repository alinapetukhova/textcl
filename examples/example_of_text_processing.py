import textcl
import pandas as pd

SOURCE_FILE_PATH = 'prepared_bbc_dataset.csv'

# getting text data from file
input_texts_df = pd.read_csv(SOURCE_FILE_PATH).reset_index()
print(input_texts_df)

# splitting texts into sentences
split_input_texts_df = textcl.split_into_sentences(input_texts_df)
print("Num sentences before filtering: {}".format(len(split_input_texts_df)))

# filtering on language
split_input_texts_df = textcl.language_filtering(split_input_texts_df, threshold=0.99, language='en')
print("Num sentences after language filtering: {}".format(len(split_input_texts_df)))

# filtering on Jaccard similarity
split_input_texts_df = textcl.jaccard_sim_filtering(split_input_texts_df, threshold=0.8)
print("Num sentences after Jaccard sim filtering: {}".format(len(split_input_texts_df)))

# filtering on perplexity score
split_input_texts_df = textcl.perplexity_filtering(split_input_texts_df, threshold=5)
print("Num sentences after perplexity filtering: {}".format(len(split_input_texts_df)))

# outliers filtering
joined_texts = split_input_texts_df[["text", "topic_name"]].drop_duplicates()
joined_texts = joined_texts[joined_texts.topic_name == 'tech']
joined_texts, _ = textcl.outlier_detection(joined_texts, method='rpca', Z_threshold=0.8)
print("Num sentences after outliers filtering: {}".format(len(input_texts_df)))
print(joined_texts)

# joining sentences into texts
joined_texts = textcl.join_sentences_by_label(split_input_texts_df)
print("Num texts after joining: {}".format(len(joined_texts)))
print(joined_texts)

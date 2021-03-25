import pytest

from textcl.preprocessing import *
from textcl.outliers_detection import *
import pandas as pd


def test_perplexity_filtering():
    """
    To test filtering unmeaning sentences by perplexity filtering. Failure appears\
    if the incoherent sentence is not filtered. For example raw from the table
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'Our work contributes to these Sustainable Development Goals: Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference. From washing glassware to disposing of gloves, everything we do in our labs contributes to our environmental footprint.',
                                'Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference.'],
                               ['Climate change',
                                '11',
                                'Sustainability approach Access to healthcare Environmental protection Ethics and transparency Notices Bayer Sustainability Report 2019 36 3%?!353? 1. 5В°C 1} 33%.?',
                                'Sustainability approach Access to healthcare Environmental protection Ethics and transparency Notices Bayer Sustainability Report 2019 36 3%?!353? 1. 5В°C 1} 33%.?']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    output_data = pd.DataFrame([['Plastic reduction',
                                 '22',
                                 'Our work contributes to these Sustainable Development Goals: Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference. From washing glassware to disposing of gloves, everything we do in our labs contributes to our environmental footprint.',
                                 'Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference.']],
                               columns=['topic_name', 'document_id', 'text', 'sentence'])
    result = perplexity_filtering(input_data, threshold=5)
    assert result.equals(output_data)


def test_language_filtering():
    """
    To test sentences filtering by language. Failure appears\
    if the sentence in language different from english is not filtered
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'Our work contributes to these Sustainable Development Goals: Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference. From washing glassware to disposing of gloves, everything we do in our labs contributes to our environmental footprint.',
                                'Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference.'],
                               ['Climate change',
                                '11',
                                '行動中的可持續發展員工塑造可持續發展在全球各地，我們通過舉辦阿斯利康可持續發展週來加強對負責任行為的承諾。 每個活動都具有可持續性的不同方面為此我們強調責任制和如何採取行動。',
                                '每個活動都具有可持續性的不同方面，為此我們強調責任制和如何採取行動。']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    output_data = pd.DataFrame([['Plastic reduction',
                                 '22',
                                 'Our work contributes to these Sustainable Development Goals: Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference. From washing glassware to disposing of gloves, everything we do in our labs contributes to our environmental footprint.',
                                 'Sustainability in action Creating a culture of sustainability in the labs At Bayer, we want to innovate new medicines for patients in sustainable ways, and how we work in the lab makes a difference.']],
                               columns=['topic_name', 'document_id', 'text', 'sentence'])
    result = language_filtering(input_data, threshold=0.99, language='en')
    print(result)
    print(output_data)
    assert result.equals(output_data)


def test_jaccard_sim_filtering():
    """
    To test filtering of very similar sentences by Jaccard similarity. Failure appears\
    if the same sentences are not filtered
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'This is example of the first sentence. This is example of the second sentence',
                                'This is example of the second sentence.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence. This is example of the second sentence',
                                'This is example of the second sentence.']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    output_data = pd.DataFrame([['Climate change', '11',
                                 'This is example of the first sentence. This is example of the second sentence',
                                 'This is example of the second sentence.']],
                               columns=['topic_name', 'document_id', 'text', 'sentence'])
    result = jaccard_sim_filtering(input_data, threshold=0.8)
    assert result.equals(output_data)


def test_split_into_sentences():
    """
    To test function to split texts into sentences. Failure appears if list of\
    result sentences are not the same to the expected list
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'This is example of the first sentence. This is example of the second sentence'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence. This is example of the second sentence']],
                              columns=['topic_name', 'document_id', 'text'])
    output_data = pd.DataFrame([['Plastic reduction',
                                 '22',
                                 'This is example of the first sentence. This is example of the second sentence',
                                 'This is example of the first sentence.'],
                                ['Plastic reduction',
                                 '22',
                                 'This is example of the first sentence. This is example of the second sentence',
                                 'This is example of the second sentence'],
                                ['Climate change',
                                 '11',
                                 'This is example of the first sentence. This is example of the second sentence',
                                 'This is example of the first sentence.'],
                                ['Climate change',
                                 '11',
                                 'This is example of the first sentence. This is example of the second sentence',
                                 'This is example of the second sentence']],
                               columns=['topic_name', 'document_id', 'text', 'sentence'])
    result = split_into_sentences(input_data)
    assert result.equals(output_data)


def test_outliers_detection_result_len():
    """
    To compare the number of output sentences from test_outliers_detection function with number of input sentences.\
    Failure appears if the number of input sentences isn't match sum number of outliers sentences and normal sentences
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the first sentence of the first text.'],
                               ['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the second sentence of the first text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the first sentence of the second text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the second sentence of the second text.']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    normal_df, outlier_df = outlier_detection(input_data, method='rpca')
    assert len(input_data) == (len(normal_df) + len(outlier_df))


def test_rpca_implementation_result_dimensions():
    """
    To test match of outlier matrix dimension resulted from RPCA method and bag of words for input sentences dimension.\
    Failure appears if dimensions are not the same
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the first sentence of the first text.'],
                               ['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the second sentence of the first text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the first sentence of the second text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the second sentence of the second text.']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(input_data['sentence']).todense()
    result_matrix = rpca_implementation(bag_of_words)
    print(len(input_data))
    print(len(result_matrix))
    print(bag_of_words.shape)
    assert bag_of_words.shape == result_matrix.shape


def test_tonmf_result_dimensions():
    """
    To test match of outlier matrix dimension resulted from TONMF method and bag of words for input\
    sentences dimension. Failure appears if dimensions are not the same
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the first sentence of the first text.'],
                               ['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the second sentence of the first text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the first sentence of the second text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the second sentence of the second text.']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(input_data['sentence']).todense()
    result_matrix, _, _, _ = tonmf(bag_of_words, k=3, alpha=1, beta=0.5)
    print(len(input_data))
    print(len(result_matrix))
    print(bag_of_words.shape)
    assert bag_of_words.shape == result_matrix.shape


def test_svd_result_dimensions():
    """
    To test match of outlier matrix dimension resulted from SVD method and bag of words for input sentences dimension.\
    Failure appears if dimensions are not the same
    """

    input_data = pd.DataFrame([['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the first sentence of the first text.'],
                               ['Plastic reduction',
                                '22',
                                'This is example of the first sentence of the first text. This is example of the second sentence',
                                'This is example of the second sentence of the first text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the first sentence of the second text.'],
                               ['Climate change',
                                '11',
                                'This is example of the first sentence of the second text. This is example of the second sentence',
                                'This is example of the second sentence of the second text.']],
                              columns=['topic_name', 'document_id', 'text', 'sentence'])
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(input_data['sentence']).todense()
    result_matrix = svd(bag_of_words)
    print(len(input_data))
    print(len(result_matrix))
    print(bag_of_words.shape)
    assert bag_of_words.shape == result_matrix.shape

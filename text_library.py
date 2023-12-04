"""
filename: textastic.py
description: An extensible reusable library for text analysis and comparison
"""

from collections import defaultdict, Counter
import random as rnd
import matplotlib.pyplot as plt
import string
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
from plotly.subplots import make_subplots


class ParserError(Exception):
    """Raised when parser is given incompatible files"""
    pass


class Textastic:

    # Defines class attribute to contain stopwords, initialized as empty set
    stop_words = set()

    def __init__(self):
        # string  --> {filename/label --> statistics}
        # "wordcounts" --> {"A": wc_A, "B": wc_B, ....}

        self.data = defaultdict(dict)

    def _save_results(self, label, results):
        for k, v in results.items():
            self.data[k][label] = v

    @staticmethod
    def _sentiment_analysis(lines):

        # Loads in pre-defined positive and negative words
        pos_file = open('positive-words.txt', 'r')
        pos = set(pos_file.read().split())

        neg_file = open('negative-words.txt', 'r')
        neg = set(neg_file.read().split())

        pos_ct = 0
        neg_ct = 0
        total_ct = len(lines)

        # Finds count of pos / neg sentiment words in the document
        for word in lines:
            if word in pos:
                pos_ct += 1
            elif word in neg:
                neg_ct += 1

        # Calculates percentage of positive words to total words
        score = (pos_ct - neg_ct) / total_ct
        return score

    @staticmethod
    def _split(filename):
        """Returns the value for which to split the file"""

        try:
            # Determines file format and returns split value
            if filename.endswith('.txt'):
                return ' '
            elif filename.endswith('.csv'):
                return ', '
            else:
                raise ParserError

        except ParserError:
            print('Please provide txt or csv files')


    @staticmethod
    def _default_parser(filename):
        """Removes whitespace, punctuation, and capitalization"""

        # Opens and reads file, finding split value from format
        split = Textastic._split(filename)
        f = open(filename, 'r')
        text = f.read()
        f.close()
        lines = text.split(split)

        # Removes punctuation, capitalization, and splits text, removing stopwords
        translator = str.maketrans("", "", string.punctuation)
        lines = [word.translate(translator).lower() for word in lines]
        lines = [word for word in lines if word not in Textastic.stop_words]

        # Calculates the number of words
        num_words = 0
        num_words += len(lines)

        # Calculates sentiment score
        sentiment = Textastic._sentiment_analysis(lines)

        # Assigns results and returns statistics
        results = {
            'wordcount': Counter(lines),
            'numwords': num_words,
            'sentiment': sentiment
        }
        return results

    def load_text(self, filename, label=None, parser=None):
        """ Registers a text document with the framework and processes data"""

        if parser is None:
            results = Textastic._default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        # Stores the results of processing one file in the internal state data
        self._save_results(label, results)

    @staticmethod
    def load_stop_words(stopfile):
        """Filters out common / stopwords from the text"""
        # Reads file and assigns stopword set to class variable
        f = open(stopfile, 'r')
        split = Textastic._split(stopfile)
        text = f.read()
        f.close()
        lines = text.split(split)

        # Removes punctuation, capitalization, and splits text, removing stopwords
        translator = str.maketrans("", "", string.punctuation)
        lines = [word.translate(translator).lower() for word in lines]
        lines = [word.strip('\n') for word in lines]
        stop_set = set([word for word in lines if word not in Textastic.stop_words])
        Textastic.stop_words = stop_set

    def compare_num_words(self):
        """Bar chart comparing number of words in each file"""
        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.show()

    def viz2(self, top_n=15):
        """Creates subplots from k most common words, interactive"""
        # Extract data for the subplots
        subplot_data = self.data['wordcount']

        # Create subplots with one row per subplot and one column
        num_subplots = len(subplot_data)
        fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=list(subplot_data.keys()))

        # Iterate over each subplot and add a bar trace
        subplot_num = 1
        for label, wordcount in subplot_data.items():
            # Sort the wordcount dictionary by values in descending order and take the top N
            top_words = dict(sorted(wordcount.items(), key=lambda x: x[1], reverse=True)[:top_n])

            labels = [f'{word}' for word in top_words.keys()]
            values = list(top_words.values())

            fig.add_trace(
                go.Bar(x=labels, y=values, name=label),
                row=subplot_num, col=1
            )
            subplot_num += 1

        # Update layout for subplots
        fig.update_layout(height=400 * num_subplots, showlegend=False)

        # Update layout for the bar chart trace
        fig.update_layout(autosize=False)

        return dcc.Graph(
            id='bar-graph',
            figure=fig,
            style={'height': 'auto'}
        )

    def viz3(self):
        """Creates bar graph based off sentiment scores for all documents"""

        # Extract data for the bar graph
        labels, values = [], []

        for label, wordcount in self.data['sentiment'].items():
            labels.append(label)
            values.append(wordcount)

        return dcc.Graph(
            id='bar-graph-unique-words',
            figure={
                'data': [
                    {'x': labels, 'y': values, 'type': 'bar', 'name': 'Documents'},
                ],
                'layout': {
                    'title': 'Sentiment Score per Document',
                    'xaxis': {'title': 'Documents'},
                    'yaxis': {'title': 'Sentiment Score'},
                }
            },
            style={'height': '400px'}  # Fixed height for the graph
        )

    def combined_viz(self, top_n=15):
        """Runs a combined interactive Dashboard containing relevant visualizations"""
        app = Dash(__name__)

        app.layout = html.Div([
            html.H1('Natural Language Processor - Combined Viz'),
            dcc.Tabs([
                dcc.Tab(label='Viz1 : Sankey', children=[self.make_sankey()]),
                dcc.Tab(label='Viz2 : K Most Common Words', children=[
                    dcc.Slider(
                        id='top-n-slider',
                        min=0,
                        max=15,
                        step=1,
                        marks={i: str(i) for i in range(0, 16, 1)},
                        value=top_n
                    ),
                    dcc.Store(id='viz2-store', data={'top_n': top_n}),
                    dcc.Graph(id='bar-graph')
                ]),
                dcc.Tab(label='Viz3 : Sentiment', children=[self.viz3()])
            ], style={'height': 'auto'}),
        ])

        # Callback to update viz2 figure based on the Slider value
        @app.callback(
            Output('viz2-store', 'data'),
            Input('top-n-slider', 'value')
        )
        def update_top_n(top_n_value):
            return {'top_n': top_n_value}

        @app.callback(
            Output('bar-graph', 'figure'),
            Input('viz2-store', 'data')
        )
        def update_viz2(data):
            top_n = data['top_n']

            # Extract data for the subplots
            subplot_data = self.data['wordcount']

            # Create subplots with one row per subplot and one column
            num_subplots = len(subplot_data)
            fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=list(subplot_data.keys()))

            # Iterate over each subplot and add a bar trace
            subplot_num = 1
            for label, wordcount in subplot_data.items():
                # Sort the wordcount dictionary by values in descending order and take the top N
                top_words = dict(sorted(wordcount.items(), key=lambda x: x[1], reverse=True)[:top_n])

                labels = [f'Words - {word}' for word in top_words.keys()]
                values = list(top_words.values())

                fig.add_trace(
                    go.Bar(x=labels, y=values, name=label),
                    row=subplot_num, col=1
                )
                subplot_num += 1

            # Update layout for subplots
            fig.update_layout(height=400 * num_subplots, showlegend=False)

            # Update layout for the bar chart trace
            fig.update_layout(autosize=False)

            return {'data': fig['data'], 'layout': fig['layout']}

        # Run the app
        app.run_server(debug=True)

    def make_sankey(self, specified_set=None, k=5):
        """Creates sankey diagram from word counts"""
        # Creates labels from keys in wordcount
        text_labels = list(self.data['wordcount'].keys())

        # Defines lists to contain sankey variables
        long_sources = []
        long_targets = []
        long_values = []
        long_labels = [text_label for text_label in text_labels]

        # For each text and index in loaded texts
        for i, text in enumerate(text_labels):

            # Sorts words by frequency
            dict_copy = sorted(self.data['wordcount'][text].items(), key=lambda i: i[1], reverse=True)
            word_list = [pair[0] for pair in dict_copy]

            if specified_set is not None:
                # Removes irrelevant words from word list
                word_list = [word for word in word_list if word in specified_set]

            if k is not None:
                # Reduces word_list down to k OR length of list if k > len to prevent over-indexing
                word_list = word_list[: (min(k, len(word_list)))]

            # For each word in the parsed / processed document
            for word in word_list:
                long_sources.append(i)
                long_labels.append(word)
                long_targets.append(long_labels.index(word))
                long_values.append(self.data['wordcount'][text][word])

        # Creates figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=long_labels,
            ),
            link=dict(
                source=long_sources,
                target=long_targets,
                value=long_values
            ))])

        fig.show()

        return dcc.Graph(
            id='sankey',
            figure=fig,
            style={'height': 'auto'}
        )

